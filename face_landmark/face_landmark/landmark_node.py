#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

from insightface.app import FaceAnalysis


class InsightFaceLandmarkNode(Node):
    def __init__(self):
        super().__init__('insightface_landmark_node')

        # ===== Parameters =====
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('annotated_topic', '/face/annotated')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('det_size', 640)   # 640 or 1024 추천
        self.declare_parameter('draw_radius', 1)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        annotated_topic = self.get_parameter('annotated_topic').get_parameter_value().string_value
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        det_size = self.get_parameter('det_size').get_parameter_value().integer_value

        self.draw_radius = int(self.get_parameter('draw_radius').value)

        # ===== CV Bridge =====
        self.bridge = CvBridge()

        # ===== InsightFace =====
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.app = FaceAnalysis(providers=providers)
        ctx_id = 0 if use_gpu else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

        self.get_logger().info(f'InsightFace providers: {providers}, ctx_id={ctx_id}, det_size={det_size}')

        # ===== ROS I/O =====
        self.sub = self.create_subscription(Image, image_topic, self.image_cb, 10)
        self.pub = self.create_publisher(Image, annotated_topic, 10)

        self.get_logger().info(f'Subscribing: {image_topic}')
        self.get_logger().info(f'Publishing:  {annotated_topic}')

    def image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        img_visual = bgr.copy()

        try:
            faces = self.app.get(bgr)
        except Exception as e:
            self.get_logger().error(f'InsightFace inference failed: {e}')
            return

        # draw landmarks
        for face in faces:
            if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
                continue
            lm = face.landmark_2d_106.astype(np.int32)
            for (x, y) in lm:
                cv2.circle(img_visual, (int(x), int(y)), self.draw_radius, (0, 0, 255), -1)

            # optional: bbox
            if hasattr(face, 'bbox') and face.bbox is not None:
                x1, y1, x2, y2 = face.bbox.astype(np.int32)
                cv2.rectangle(img_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out_msg = self.bridge.cv2_to_imgmsg(img_visual, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = InsightFaceLandmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
