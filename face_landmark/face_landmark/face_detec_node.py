#!/usr/bin/env python3
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Action Interface
from inha_interfaces.action import WaitPerson
from insightface.app import FaceAnalysis

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        # 설정
        self.declare_parameter('img_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('gpu_device', 0)
        
        self.img_topic = self.get_parameter('img_topic').value
        use_gpu = self.get_parameter('use_gpu').value
        gpu_device = self.get_parameter('gpu_device').value

        # AI 모델 로드 (Detection 전용)
        # rec_name=None을 주면 인식 모델을 로드하지 않아 가벼워짐
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(allowed_modules=['detection'], providers=providers)
        self.face_app.prepare(ctx_id=gpu_device if use_gpu else -1, det_size=(640, 640))

        # 통신
        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()
        self.sub_img = self.create_subscription(
            Image, self.img_topic, self.img_callback, 10, callback_group=self.cb_group
        )

        # Action Server
        self._action_server = ActionServer(
            self,
            WaitPerson,
            'wait_person',
            self.execute_callback,
            callback_group=self.cb_group
        )

        self.last_detected_time = 0.0
        self.get_logger().info('Detection Node (WaitPerson) Ready.')

    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 얼굴 감지 수행
            faces = self.face_app.get(img)
            
            # 얼굴이 있고, 크기가 어느정도 클 때 (노이즈 방지)
            if len(faces) > 0:
                # 가장 큰 얼굴 확인
                max_area = 0
                for f in faces:
                    w = f.bbox[2] - f.bbox[0]
                    h = f.bbox[3] - f.bbox[1]
                    if w * h > max_area:
                        max_area = w * h
                
                # 최소 크기 조건 (예: 60x60 이상)
                if max_area > (60 * 60):
                    self.last_detected_time = time.time()
                    
        except Exception as e:
            self.get_logger().warn(f'Image processing error: {e}')

    def execute_callback(self, goal_handle):
        self.get_logger().info('WaitPerson Action Requested')
        goal = goal_handle.request
        feedback = WaitPerson.Feedback()
        result = WaitPerson.Result()

        start_time = time.time()
        timeout_sec = goal.timeout_ms / 1000.0
        
        # 시작 전 기존 감지 기록 초기화
        self.last_detected_time = 0.0

        while (time.time() - start_time) < timeout_sec:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return result

            # 최근 0.5초 내에 감지 기록이 있으면 성공
            if (time.time() - self.last_detected_time) < 0.5:
                feedback.state = "DETECTED"
                goal_handle.publish_feedback(feedback)
                result.success = True
                goal_handle.succeed()
                self.get_logger().info('Person Detected!')
                return result
            
            feedback.state = "WAITING"
            goal_handle.publish_feedback(feedback)
            time.sleep(0.1)

        # Timeout
        feedback.state = "TIMEOUT"
        goal_handle.publish_feedback(feedback)
        result.success = False
        goal_handle.abort()
        self.get_logger().info('WaitPerson Timeout')
        return result

def main():
    rclpy.init()
    node = DetectionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()