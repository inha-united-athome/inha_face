#!/usr/bin/env python3
import os, time, json, glob
from typing import Optional, List
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# Action Interface
from inha_interfaces.action import CaptureFaceCrop
from insightface.app import FaceAnalysis

class RecognitionNode(Node):
    def __init__(self):
        super().__init__('recognition_node') 

        # -------- 파라미터 (기존 코드 유지) --------
        self.declare_parameter('img_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('out_topic', '/face/analysis')
        self.declare_parameter('vis_topic', '/face/vis')
        self.declare_parameter('db_path', '/home/user/face_data')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('gpu_device', 0)
        self.declare_parameter('recog_threshold', 0.45)
        
        self.img_topic = self.get_parameter('img_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.vis_topic = self.get_parameter('vis_topic').value
        self.db_path = self.get_parameter('db_path').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.gpu_device = self.get_parameter('gpu_device').value
        self.recog_threshold = self.get_parameter('recog_threshold').value

        os.makedirs(self.db_path, exist_ok=True)

        # -------- 통신 및 AI --------
        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        # 모델 로드 (Detection + Recognition)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.face_app.prepare(ctx_id=self.gpu_device if self.use_gpu else -1, det_size=(640, 640))

        # 갤러리 로드
        self.db_names = []
        self.db_embs = []
        self._load_gallery()

        # Subscriber / Publisher
        self.sub_img = self.create_subscription(Image, self.img_topic, self.img_callback, 10, callback_group=self.cb_group)
        self.pub_json = self.create_publisher(String, self.out_topic, 10)
        self.pub_vis = self.create_publisher(Image, self.vis_topic, 10)

        # Action Server (CaptureFaceCrop)
        self._action_server = ActionServer(
            self,
            CaptureFaceCrop,
            'capture_face_crop',
            self.execute_capture,
            callback_group=self.cb_group
        )

        # 캡처 제어 변수
        self.capture_active = False
        self.capture_info = {} # {name, count, target, save_dir}
        
        self.get_logger().info('Recognition Node (CaptureFaceCrop) Ready.')

    # -------- 갤러리 로드 함수 --------
    def _load_gallery(self):
        """DB 폴더에서 얼굴 임베딩 로드"""
        self.db_names = []
        self.db_embs = []
        
        if not os.path.exists(self.db_path):
            self.get_logger().warn(f'DB 경로 없음: {self.db_path}')
            return
        
        # 각 사람 폴더 순회
        for person_name in os.listdir(self.db_path):
            person_dir = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            # 이미지 파일들 로드
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
            
            # 평균 임베딩 계산
            if embeddings:
                avg_emb = np.mean(embeddings, axis=0)
                self.db_names.append(person_name)
                self.db_embs.append(avg_emb)
                self.get_logger().info(f'로드됨: {person_name} ({len(embeddings)}장)')
        
        self.get_logger().info(f'갤러리 로드 완료: {len(self.db_names)}명')

    # -------- 메인 콜백 --------
    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # AI 처리
            faces = self.face_app.get(img)
            vis = img.copy()
            faces_out = []

            # 갤러리 비교용 데이터
            db_embs_np = np.stack(self.db_embs, axis=0) if self.db_embs else None

            largest_face_crop = None
            max_area = 0

            for f in faces:
                bbox = f.bbox.astype(int)
                w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                
                # 가장 큰 얼굴 찾기 (캡처용)
                if w*h > max_area:
                    max_area = w*h
                    largest_face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # 인식 (Recognition)
                name = "Unknown"
                sim = 0.0
                if f.embedding is not None and db_embs_np is not None:
                    # Cosine Similarity 계산
                    e = f.embedding / np.linalg.norm(f.embedding)
                    g = db_embs_np / np.linalg.norm(db_embs_np, axis=1, keepdims=True)
                    sims = np.dot(g, e)
                    idx = np.argmax(sims)
                    if sims[idx] > self.recog_threshold:
                        name = self.db_names[idx]
                        sim = float(sims[idx])

                # 시각화 및 데이터
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                cv2.putText(vis, f"{name} ({sim:.2f})", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                faces_out.append({'name': name, 'bbox': bbox.tolist(), 'sim': sim})

            # --- [Capture Action 로직] ---
            if self.capture_active and largest_face_crop is not None and largest_face_crop.size > 0:
                info = self.capture_info
                if info['count'] < info['target']:
                    # 이미지 저장
                    ts = int(time.time() * 1000)
                    path = os.path.join(info['save_dir'], f"{ts}.jpg")
                    cv2.imwrite(path, largest_face_crop)
                    
                    info['count'] += 1
                    self.get_logger().info(f"Captured {info['count']}/{info['target']}")
                    
                    # 캡처 속도 조절
                    time.sleep(0.2)

            # Publish
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
            self.pub_json.publish(String(data=json.dumps(faces_out)))

        except Exception as e:
            self.get_logger().error(f"Img CB Error: {e}")

    # -------- Action Server: CaptureFaceCrop --------
    def execute_capture(self, goal_handle):
        goal = goal_handle.request
        self.get_logger().info(f"Start Capturing for {goal.person_name}")
        
        save_dir = os.path.join(self.db_path, goal.person_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 캡처 모드 활성화
        self.capture_info = {
            'target': goal.num_images,
            'count': 0,
            'save_dir': save_dir,
            'name': goal.person_name
        }
        self.capture_active = True
        
        feedback = CaptureFaceCrop.Feedback()
        result = CaptureFaceCrop.Result()
        
        start_time = time.time()
        timeout = goal.timeout_ms / 1000.0

        while self.capture_info['count'] < goal.num_images:
            if time.time() - start_time > timeout:
                self.capture_active = False
                result.success = False
                result.message = "Timeout"
                goal_handle.abort()
                return result
            
            if goal_handle.is_cancel_requested:
                self.capture_active = False
                goal_handle.canceled()
                return result

            feedback.state = "CAPTURING"
            goal_handle.publish_feedback(feedback)
            time.sleep(0.1)

        # 완료
        self.capture_active = False
        
        # 메모리 갤러리에 즉시 반영 (선택사항: 다시 로드하거나 추가)
        # self._load_gallery() 
        
        result.success = True
        result.saved_dir = save_dir
        result.message = "Success"
        goal_handle.succeed()
        return result

def main():
    rclpy.init()
    node = RecognitionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()