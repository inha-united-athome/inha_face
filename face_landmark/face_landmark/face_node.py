#!/usr/bin/env python3
import os
import time
import json
import glob
import threading
from collections import deque
from typing import Optional, List
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from cv_bridge import CvBridge

# Action Interface
from inha_interfaces.action import CaptureFaceCrop, WaitPerson
from insightface.app import FaceAnalysis


class FaceNode(Node):
    """
    통합 얼굴 노드
    - WaitPerson Action: 안정적인 얼굴 감지 대기
    - CaptureFaceCrop Action: 얼굴 크롭 이미지 저장
    - 실시간 얼굴 인식 및 시각화
    """
    
    def __init__(self):
        super().__init__('face_node')

        # ======== 파라미터 ========
        self.declare_parameter('img_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('out_topic', '/face/analysis')
        self.declare_parameter('vis_topic', '/face/vis')
        self.declare_parameter('db_path', '/root/face_ws/src/face_data')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('gpu_device', 0)
        self.declare_parameter('recog_threshold', 0.45)
        self.declare_parameter('stable_threshold', 150.0)   # 30 → 80 (더 여유있게)
        self.declare_parameter('stable_duration', 0.8)     # 1.5 → 0.8초 (더 짧게)
        self.declare_parameter('min_face_size', 30)        # 100 → 80 (작은 얼굴도 허용)

        self.img_topic = self.get_parameter('img_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.vis_topic = self.get_parameter('vis_topic').value
        self.db_path = self.get_parameter('db_path').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.gpu_device = self.get_parameter('gpu_device').value
        self.recog_threshold = self.get_parameter('recog_threshold').value
        self.stable_threshold = self.get_parameter('stable_threshold').value
        self.stable_duration = self.get_parameter('stable_duration').value
        self.min_face_size = self.get_parameter('min_face_size').value

        os.makedirs(self.db_path, exist_ok=True)

        # ======== AI 모델 ========
        self.bridge = CvBridge()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.face_app.prepare(ctx_id=self.gpu_device if self.use_gpu else -1, det_size=(640, 640))

        # ======== 갤러리 (인식용 DB) ========
        self.db_names = []
        self.db_embs = []
        self._load_gallery()

        # ======== 상태 변수 ========
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # WaitPerson용 안정성 추적
        self.bbox_history = deque(maxlen=30)
        self.stable_start_time = None
        
        # CaptureFaceCrop용
        self.capture_active = False
        self.capture_info = {}

        # ======== 콜백 그룹 ========
        self.sub_cb_group = MutuallyExclusiveCallbackGroup()
        self.action_cb_group = ReentrantCallbackGroup()

        # ======== Subscriber / Publisher ========
        self.sub_img = self.create_subscription(
            CompressedImage, self.img_topic, self.img_callback, 10,
            callback_group=self.sub_cb_group
        )
        self.pub_json = self.create_publisher(String, self.out_topic, 10)
        self.pub_vis = self.create_publisher(Image, self.vis_topic, 10)

        # ======== Action Servers ========
        # 1. WaitPerson
        self._wait_action_server = ActionServer(
            self,
            WaitPerson,
            'wait_person',
            self.execute_wait_person,
            callback_group=self.action_cb_group
        )

        # 2. CaptureFaceCrop
        self._capture_action_server = ActionServer(
            self,
            CaptureFaceCrop,
            'capture_face_crop',
            self.execute_capture,
            callback_group=self.action_cb_group
        )

        self.get_logger().info('=== FaceNode Ready ===')
        self.get_logger().info(f'  - DB Path: {self.db_path}')
        self.get_logger().info(f'  - Actions: /wait_person, /capture_face_crop')
        self.get_logger().info(f'  - Topics: {self.out_topic}, {self.vis_topic}')

    # ================================================================
    #                       갤러리 관리
    # ================================================================
    def _load_gallery(self):
        """DB 폴더에서 얼굴 임베딩 로드"""
        self.db_names = []
        self.db_embs = []

        if not os.path.exists(self.db_path):
            self.get_logger().warn(f'DB 경로 없음: {self.db_path}')
            return

        for person_name in os.listdir(self.db_path):
            person_dir = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            image_files = glob.glob(os.path.join(person_dir, '*.jpg')) + \
                          glob.glob(os.path.join(person_dir, '*.png'))

            embeddings = []
            for img_path in image_files:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    faces = self.face_app.get(img)
                    if faces and faces[0].embedding is not None:
                        embeddings.append(faces[0].embedding)
                except Exception as e:
                    self.get_logger().warn(f'임베딩 추출 실패: {img_path}, {e}')

            if embeddings:
                avg_emb = np.mean(embeddings, axis=0)
                self.db_names.append(person_name)
                self.db_embs.append(avg_emb)
                self.get_logger().info(f'로드됨: {person_name} ({len(embeddings)}장)')

        self.get_logger().info(f'갤러리 로드 완료: {len(self.db_names)}명')

    # ================================================================
    #                       이미지 콜백
    # ================================================================
    def img_callback(self, msg):
        """카메라 압축 이미지 콜백 - 인식 + 시각화"""
        try:
            # CompressedImage -> cv2 BGR 디코딩
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn("CompressedImage 디코딩 실패 (img=None)")
                return

            # 현재 프레임 저장 (Action에서 사용)
            with self.frame_lock:
                self.current_frame = img.copy()

            # AI 처리
            faces = self.face_app.get(img)
            vis = img.copy()
            faces_out = []

            db_embs_np = np.stack(self.db_embs, axis=0) if self.db_embs else None

            for f in faces:
                bbox = f.bbox.astype(int)

                # 인식 (Recognition)
                name = "Unknown"
                sim = 0.0
                if f.embedding is not None and db_embs_np is not None:
                    e = f.embedding / np.linalg.norm(f.embedding)
                    g = db_embs_np / np.linalg.norm(db_embs_np, axis=1, keepdims=True)
                    sims = np.dot(g, e)
                    idx = np.argmax(sims)
                    if sims[idx] > self.recog_threshold:
                        name = self.db_names[idx]
                        sim = float(sims[idx])

                # 시각화
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(vis, f"{name} ({sim:.2f})", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                faces_out.append({'name': name, 'bbox': bbox.tolist(), 'sim': sim})

            # Publish (vis는 raw Image로 내보내도 OK)
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
            self.pub_json.publish(String(data=json.dumps(faces_out)))

        except Exception as e:
            self.get_logger().error(f"Img CB Error: {e}")

    # ================================================================
    #                  WaitPerson Action
    # ================================================================
    def _get_largest_face(self, frame):
        """가장 큰 얼굴 반환"""
        faces = self.face_app.get(frame)
        if not faces:
            return None

        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        width = largest.bbox[2] - largest.bbox[0]
        height = largest.bbox[3] - largest.bbox[1]
        if width < self.min_face_size or height < self.min_face_size:
            return None

        return largest

    def _check_stability(self, bbox):
        """얼굴 위치 안정성 체크 (더 관대한 버전)"""
        self.bbox_history.append(bbox)

        # 최소 5개만 있어도 체크 시작 (10 → 5)
        if len(self.bbox_history) < 5:
            return False

        recent = list(self.bbox_history)[-5:]
        
        # 첫 bbox와 마지막 bbox의 중심점 비교 (더 간단한 방식)
        first = np.array(recent[0])
        last = np.array(recent[-1])
        
        # 중심점 계산
        first_center = np.array([(first[0] + first[2]) / 2, (first[1] + first[3]) / 2])
        last_center = np.array([(last[0] + last[2]) / 2, (last[1] + last[3]) / 2])
        
        # 중심점 이동 거리
        center_diff = np.linalg.norm(last_center - first_center)

        if center_diff < self.stable_threshold:
            if self.stable_start_time is None:
                self.stable_start_time = time.time()

            elapsed = time.time() - self.stable_start_time
            if elapsed >= self.stable_duration:
                return True
        else:
            self.stable_start_time = None

        return False

    def execute_wait_person(self, goal_handle):
        """WaitPerson Action 실행 - 안정적인 얼굴 감지 대기"""
        self.get_logger().info('WaitPerson Action 시작')

        timeout_ms = goal_handle.request.timeout_ms
        timeout_sec = timeout_ms / 1000.0 if timeout_ms > 0 else 30.0

        feedback = WaitPerson.Feedback()
        result = WaitPerson.Result()

        start_time = time.time()
        self.bbox_history.clear()
        self.stable_start_time = None

        while rclpy.ok():
            elapsed = time.time() - start_time

            # 타임아웃 체크
            if elapsed > timeout_sec:
                self.get_logger().warn('타임아웃: 안정적인 얼굴을 찾지 못함')
                feedback.state = 'TIMEOUT'
                goal_handle.publish_feedback(feedback)
                goal_handle.abort()
                result.success = False
                return result

            # 취소 체크
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                return result

            # 프레임 가져오기
            with self.frame_lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None

            if frame is None:
                feedback.state = 'WAITING_FRAME'
                goal_handle.publish_feedback(feedback)
                time.sleep(0.1)
                continue

            # 얼굴 감지
            face = self._get_largest_face(frame)

            if face is None:
                feedback.state = 'SEARCHING'
                self.bbox_history.clear()
                self.stable_start_time = None
            else:
                bbox = face.bbox.tolist()

                if self._check_stability(bbox):
                    feedback.state = 'FOUND_STABLE'
                    goal_handle.publish_feedback(feedback)

                    self.get_logger().info('안정적인 얼굴 감지 완료!')
                    goal_handle.succeed()
                    result.success = True
                    return result
                else:
                    feedback.state = 'STABILIZING'

            goal_handle.publish_feedback(feedback)
            time.sleep(0.1)

        result.success = False
        return result

    # ================================================================
    #                  CaptureFaceCrop Action
    # ================================================================
    def execute_capture(self, goal_handle):
        """CaptureFaceCrop Action 실행 - 얼굴 크롭 이미지 저장"""
        goal = goal_handle.request
        self.get_logger().info(f"CaptureFaceCrop 시작: {goal.person_name}")

        save_dir = os.path.join(self.db_path, goal.person_name)
        os.makedirs(save_dir, exist_ok=True)

        feedback = CaptureFaceCrop.Feedback()
        result = CaptureFaceCrop.Result()

        start_time = time.time()
        timeout = goal.timeout_ms / 1000.0
        captured_count = 0
        target = goal.num_images

        while captured_count < target:
            # 타임아웃 체크
            if time.time() - start_time > timeout:
                result.success = False
                result.saved_dir = save_dir
                result.message = f"Timeout: captured {captured_count}/{target}"
                goal_handle.abort()
                return result

            # 취소 체크
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = "Canceled"
                return result

            # 현재 프레임 가져오기
            with self.frame_lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None

            if frame is None:
                feedback.state = "WAITING_FRAME"
                goal_handle.publish_feedback(feedback)
                time.sleep(0.1)
                continue

            # 얼굴 감지
            faces = self.face_app.get(frame)

            if not faces:
                feedback.state = "NO_FACE"
                goal_handle.publish_feedback(feedback)
                time.sleep(0.1)
                continue

            # 가장 큰 얼굴 찾기
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            bbox = largest.bbox.astype(int)

            # 얼굴 크롭 (마진 20%)
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                time.sleep(0.1)
                continue

            # 저장
            ts = int(time.time() * 1000)
            img_path = os.path.join(save_dir, f"{goal.person_name}_{captured_count:04d}_{ts}.jpg")
            cv2.imwrite(img_path, face_crop)

            captured_count += 1
            self.get_logger().info(f"캡처 {captured_count}/{target}: {img_path}")

            feedback.state = f"CAPTURING ({captured_count}/{target})"
            goal_handle.publish_feedback(feedback)

            time.sleep(0.3)

        # 완료 - 갤러리 다시 로드
        self._load_gallery()

        result.success = True
        result.saved_dir = save_dir
        result.message = f"Success: {captured_count} images saved"
        goal_handle.succeed()

        self.get_logger().info(f"캡처 완료: {save_dir}")
        return result


def main():
    rclpy.init()
    node = FaceNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
