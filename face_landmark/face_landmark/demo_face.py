#!/usr/bin/env python3
"""
Face Bag Node - rosbag에서 얼굴 데이터 추출용 노드
- 토픽으로 이름 받으면 현재 얼굴 10장 캡처하여 face_data_bag에 저장
- /face/enable 토픽으로 인식 on/off
- 인식 기능 유지 (갤러리 로드 + 실시간 인식)
"""

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
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

from insightface.app import FaceAnalysis


class FaceBagNode(Node):
    """
    Bag 추출용 얼굴 노드
    - 토픽으로 이름 수신 시 10장 캡처
    - /face/enable로 인식 on/off
    - 실시간 얼굴 인식 및 시각화
    """
    
    def __init__(self):
        super().__init__('face_bag_node')

        # ======== 파라미터 ========
        self.declare_parameter('img_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('out_topic', '/face/analysis')
        self.declare_parameter('vis_topic', '/face/vis')
        self.declare_parameter('register_topic', '/face/register_name')
        self.declare_parameter('db_path', '/root/face_ws/src/face_data_bag')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('gpu_device', 0)
        self.declare_parameter('recog_threshold', 0.45)
        self.declare_parameter('num_capture_images', 10)
        self.declare_parameter('min_face_size', 30)

        self.img_topic = self.get_parameter('img_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.vis_topic = self.get_parameter('vis_topic').value
        self.register_topic = self.get_parameter('register_topic').value
        self.db_path = self.get_parameter('db_path').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.gpu_device = self.get_parameter('gpu_device').value
        self.recog_threshold = self.get_parameter('recog_threshold').value
        self.num_capture_images = self.get_parameter('num_capture_images').value
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
        
        # 캡처 상태
        self.capture_active = False
        self.capture_name = ""
        self.capture_count = 0
        self.capture_lock = threading.Lock()
        
        # 인식 on/off 상태
        self.recognition_enabled = True  # 기본: 켜짐

        # 디버깅용
        self.frame_count = 0
        self.last_log_time = time.time()

        # ======== 콜백 그룹 ========
        self.sub_cb_group = MutuallyExclusiveCallbackGroup()
        self.register_cb_group = ReentrantCallbackGroup()

        # ======== Subscriber / Publisher ========
        self.sub_img = self.create_subscription(
            Image, self.img_topic, self.img_callback, 10,
            callback_group=self.sub_cb_group
        )
        
        # 이름 등록 토픽 구독
        self.sub_register = self.create_subscription(
            String, self.register_topic, self.register_callback, 10,
            callback_group=self.register_cb_group
        )
        
        # 인식 on/off 토픽 구독
        self.sub_enable = self.create_subscription(
            Bool, '/face/enable', self.enable_callback, 10,
            callback_group=self.register_cb_group
        )
        
        self.pub_json = self.create_publisher(String, self.out_topic, 10)
        self.pub_vis = self.create_publisher(Image, self.vis_topic, 10)

        self.get_logger().info('=== FaceBagNode Ready ===')
        self.get_logger().info(f'  - Image Topic: {self.img_topic}')
        self.get_logger().info(f'  - DB Path: {self.db_path}')
        self.get_logger().info(f'  - Register Topic: {self.register_topic}')
        self.get_logger().info(f'  - Enable Topic: /face/enable (Bool)')
        self.get_logger().info(f'  - Capture Count: {self.num_capture_images}장')
        self.get_logger().info(f'  - Recognition: {"ON" if self.recognition_enabled else "OFF (Unknown mode)"}')

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
    #                  토픽 콜백
    # ================================================================
    def register_callback(self, msg):
        """이름 토픽 수신 시 10장 캡처 시작"""
        person_name = msg.data.strip()
        
        if not person_name:
            self.get_logger().warn('빈 이름 수신, 무시함')
            return
            
        with self.capture_lock:
            if self.capture_active:
                self.get_logger().warn(f'이미 캡처 중: {self.capture_name} ({self.capture_count}/{self.num_capture_images})')
                return
            
            self.capture_active = True
            self.capture_name = person_name
            self.capture_count = 0
            
        self.get_logger().info(f'=== 캡처 시작: {person_name} ({self.num_capture_images}장) ===')

    def enable_callback(self, msg):
        """인식 on/off 토글"""
        self.recognition_enabled = msg.data
        status = "ON" if msg.data else "OFF (Unknown mode)"
        self.get_logger().info(f'=== 인식 모드: {status} ===')

    # ================================================================
    #                       이미지 콜백
    # ================================================================
    def img_callback(self, msg):
        """카메라 이미지 콜백 - 인식 + 시각화 + 캡처"""
        try:
            # 디버깅: 프레임 카운터
            self.frame_count += 1

            # Image -> cv2 BGR 변환 (cv_bridge 사용)
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if img is None:
                self.get_logger().warn("이미지 변환 실패 (img=None)")
                return

            # 디버깅: 1초마다 상태 출력
            now = time.time()
            if now - self.last_log_time >= 1.0:
                self.get_logger().info(f'[DEBUG] frame_count={self.frame_count}, img_shape={img.shape}')
                self.last_log_time = now

            # 현재 프레임 저장
            with self.frame_lock:
                self.current_frame = img.copy()

            # AI 처리
            faces = self.face_app.get(img)
            vis = img.copy()
            faces_out = []

            db_embs_np = np.stack(self.db_embs, axis=0) if self.db_embs else None

            # 캡처 상태 확인
            with self.capture_lock:
                is_capturing = self.capture_active
                capture_name = self.capture_name
                current_count = self.capture_count

            for f in faces:
                bbox = f.bbox.astype(int)

                # 인식 (Recognition) - 인식 활성화 시에만
                name = "Unknown"
                sim = 0.0
                if self.recognition_enabled and f.embedding is not None and db_embs_np is not None:
                    e = f.embedding / np.linalg.norm(f.embedding)
                    g = db_embs_np / np.linalg.norm(db_embs_np, axis=1, keepdims=True)
                    sims = np.dot(g, e)
                    idx = np.argmax(sims)
                    max_sim = float(sims[idx])
                    
                    # 디버그: 항상 최고 유사도 출력
                    self.get_logger().info(f'[DEBUG] Best match: {self.db_names[idx]} (sim={max_sim:.3f}, threshold={self.recog_threshold})')
                    
                    if max_sim > self.recog_threshold:
                        name = self.db_names[idx]
                        sim = max_sim

                # 시각화
                if is_capturing:
                    # 캡처 중일 때는 노란색
                    color = (0, 255, 255)  # Yellow (BGR)
                    label = f"[CAPTURE {current_count}/{self.num_capture_images}] {capture_name}"
                elif name != "Unknown":
                    color = (0, 255, 0)  # Green
                    label = f"{name} ({sim:.2f})"
                else:
                    color = (0, 0, 255)  # Red
                    label = "Unknown"
                    
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(vis, label, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                faces_out.append({'name': name, 'bbox': bbox.tolist(), 'sim': sim})

            # 캡처 수행 (가장 큰 얼굴)
            if is_capturing and faces:
                self._capture_largest_face(img, faces)

            # Publish
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
            self.pub_json.publish(String(data=json.dumps(faces_out)))

        except Exception as e:
            self.get_logger().error(f"Img CB Error: {e}")

    def _capture_largest_face(self, frame, faces):
        """가장 큰 얼굴을 캡처하여 저장"""
        with self.capture_lock:
            if not self.capture_active:
                return
            if self.capture_count >= self.num_capture_images:
                # 캡처 완료
                self.capture_active = False
                self.get_logger().info(f'=== 캡처 완료: {self.capture_name} ({self.capture_count}장) ===')
                # 갤러리 다시 로드
                self._load_gallery()
                return
            
            capture_name = self.capture_name
            capture_count = self.capture_count
        
        # 가장 큰 얼굴 찾기
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = largest.bbox.astype(int)
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width < self.min_face_size or height < self.min_face_size:
            return

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
            return

        # 저장 디렉토리 생성
        save_dir = os.path.join(self.db_path, capture_name)
        os.makedirs(save_dir, exist_ok=True)

        # 저장
        ts = int(time.time() * 1000)
        img_path = os.path.join(save_dir, f"{capture_name}_{capture_count:04d}_{ts}.jpg")
        cv2.imwrite(img_path, face_crop)

        with self.capture_lock:
            self.capture_count += 1
            new_count = self.capture_count

        self.get_logger().info(f"캡처 {new_count}/{self.num_capture_images}: {img_path}")

        # 약간의 딜레이 (다양한 각도 캡처를 위해)
        time.sleep(0.2)


def main():
    rclpy.init()
    node = FaceBagNode()
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
