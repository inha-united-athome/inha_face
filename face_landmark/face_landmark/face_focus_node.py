#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceFocusNode(Node):
    def __init__(self):
        super().__init__('face_focus_node')

        # ===== 파라미터 =====
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        # [중요] RGB와 픽셀 매칭이 된 Depth 토픽이어야 합니다.
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw') 
        self.declare_parameter('annotated_topic', '/face/target_visual')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('det_size', 640)
        
        # 필터링 임계값
        self.declare_parameter('gaze_threshold', 0.4) # 0.0에 가까울수록 정면 (코가 중앙)
        self.declare_parameter('max_dist_mm', 3000)   # 3미터 이내만 인식

        # 파라미터 가져오기
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        annotated_topic = self.get_parameter('annotated_topic').value
        use_gpu = self.get_parameter('use_gpu').value
        det_size = self.get_parameter('det_size').value
        
        self.gaze_th = self.get_parameter('gaze_threshold').value
        self.max_dist = self.get_parameter('max_dist_mm').value

        # ===== 초기화 =====
        self.bridge = CvBridge()
        
        # InsightFace
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))

        # ===== ROS Subscriber =====
        # 1. RGB 구독
        self.sub_rgb = self.create_subscription(
            Image, rgb_topic, self.rgb_cb, qos_profile_sensor_data)
        
        # 2. Depth 구독
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self.depth_cb, qos_profile_sensor_data)

        self.pub = self.create_publisher(Image, annotated_topic, 10)

        # Depth 이미지 저장용 변수
        self.latest_depth_img = None

        self.get_logger().info(f'Start Focus Node. RGB: {rgb_topic}, Depth: {depth_topic}')

    def depth_cb(self, msg):
        """Depth 이미지를 받아서 최신 상태로 유지"""
        try:
            # RealSense Depth는 보통 16UC1 (mm 단위)
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_img = cv_depth
        except Exception as e:
            self.get_logger().error(f"Depth convert fail: {e}")

    def check_gaze(self, kps):
        """
        간단한 정면 응시 판별 (눈-코 비율)
        kps: 5개 랜드마크 [왼눈, 오른눈, 코, 왼입, 오른입]
        """
        # 왼쪽 눈 ~ 코 거리 (X축 기준)
        dist_l_eye_nose = abs(kps[0][0] - kps[2][0])
        # 오른쪽 눈 ~ 코 거리
        dist_r_eye_nose = abs(kps[1][0] - kps[2][0])

        # 비율 계산 (한쪽이 0이 되는 거 방지)
        if dist_l_eye_nose == 0 or dist_r_eye_nose == 0:
            return False, 99.9

        # 비율 차이 (0.0이면 완벽한 정면, 숫자가 클수록 측면)
        ratio_diff = abs(dist_l_eye_nose - dist_r_eye_nose) / max(dist_l_eye_nose, dist_r_eye_nose)
        
        is_frontal = ratio_diff < self.gaze_th
        return is_frontal, ratio_diff

    def get_face_depth(self, bbox, depth_img):
        """얼굴 중앙의 Depth 값 가져오기 (mm)"""
        if depth_img is None:
            return 0
        
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        # 이미지 범위 체크
        h, w = depth_img.shape
        if 0 <= cx < w and 0 <= cy < h:
            dist = depth_img[cy, cx]
            # 0이면 측정 불가 (구멍 뚫린 곳), 주변값 평균을 쓰거나 그냥 무시
            if dist == 0: 
                return 99999
            return dist
        return 99999

    def rgb_cb(self, msg):
        if self.latest_depth_img is None:
            # Depth가 아직 안 들어왔으면 대기
            return

        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            return

        vis_img = img_bgr.copy()
        faces = self.app.get(img_bgr)

        candidates = [] # (face_obj, distance)

        # 1. 모든 얼굴에 대해 필터링 수행
        for face in faces:
            bbox = face.bbox.astype(int)
            kps = face.kps # 5 landmarks

            # 1-1. Gaze Check (정면인가?)
            is_frontal, gaze_score = self.check_gaze(kps)

            # 1-2. Depth Check (거리는?)
            dist_mm = self.get_face_depth(bbox, self.latest_depth_img)

            # 시각화용 변수
            color = (0, 0, 255) # 기본: 빨강 (탈락)
            status_text = f"Side({gaze_score:.2f})"

            # 조건: 정면이고 + 거리 유효하고 + 최대 거리 이내
            if is_frontal and 0 < dist_mm < self.max_dist:
                candidates.append((face, dist_mm))
                color = (0, 255, 255) # 후보: 노랑
                status_text = f"Cand {dist_mm}mm"
            elif not is_frontal:
                status_text = f"Side({gaze_score:.2f})"
            elif dist_mm >= self.max_dist:
                status_text = f"Far {dist_mm}mm"

            # 그리기 (후보군/탈락군 표시)
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(vis_img, status_text, (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. 최종 타겟 선정 (후보 중 가장 가까운 사람)
        target_face = None
        if candidates:
            # 거리(x[1]) 기준으로 오름차순 정렬 -> 0번이 가장 가까움
            candidates.sort(key=lambda x: x[1])
            target_face = candidates[0][0]
            target_dist = candidates[0][1]

            # 타겟 강조 (초록색 굵은 박스)
            t_bbox = target_face.bbox.astype(int)
            cv2.rectangle(vis_img, (t_bbox[0], t_bbox[1]), (t_bbox[2], t_bbox[3]), (0, 255, 0), 4)
            cv2.putText(vis_img, f"TARGET ({target_dist}mm)", (t_bbox[0], t_bbox[1]-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 여기에 "말하는 사람 인식 로직(MAR)"을 추가하면 됩니다.

        # 결과 발행
        out_msg = self.bridge.cv2_to_imgmsg(vis_img, 'bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    node = FaceFocusNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()