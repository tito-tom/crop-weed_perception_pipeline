#!/usr/bin/env python3
"""
plant_perception_node.py — Main ROS 2 Perception Node
======================================================
Subscribes to RGB + Depth image topics, runs ONNX inference, assigns BYTETracker IDs,
samples median depth at plant root keypoints, and publishes TrackedPlantArray telemetry.
"""

import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

try:
    from plant_perception.msg import TrackedPlant, TrackedPlantArray
except ImportError:
    TrackedPlant = None
    TrackedPlantArray = None

from .onnx_inferencer import ONNXInferencer
from .postprocess import postprocess
from .tracker import Tracker
from .depth_utils import sample_depth_median
from .visualization import draw_visualizations

CLASS_NAMES = {
    0: "crop_small_leaf",
    1: "crop_large_leaf",
    2: "weed_small_leaf",
    3: "weed_large_leaf",
}


class PlantPerceptionNode(Node):
    def __init__(self):
        super().__init__("plant_perception_node")
        self.get_logger().info("Starting plant_perception_node...")

        self.declare_parameter("model_path", "yolov11-seg-root.onnx")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("device", "cpu")

        model_path = self.get_parameter("model_path").value
        imgsz = self.get_parameter("imgsz").value
        device = self.get_parameter("device").value

        self.bridge = CvBridge()
        self.inferencer = ONNXInferencer(model_path=model_path, device=device, imgsz=imgsz)
        self.tracker = Tracker()

        # Publishers
        self.pub_plants = self.create_publisher(TrackedPlantArray, "/plant_perception/tracked_plants", 10)
        self.pub_viz = self.create_publisher(Image, "/plant_perception/image_annotated", 10)

        # Synchronized Subscribers (RGB + Depth)
        sub_color = message_filters.Subscriber(self, Image, "/camera/color/image_raw")
        sub_depth = message_filters.Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw")

        self.ts = message_filters.ApproximateTimeSynchronizer([sub_color, sub_depth], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.image_callback)

    def image_callback(self, color_msg, depth_msg):
        t0 = time.time()
        cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        h_orig, w_orig = cv_color.shape[:2]

        outputs, ratio, pad, infer_ms = self.inferencer.run(cv_color)
        detections = postprocess(outputs, orig_shape=(h_orig, w_orig), num_classes=4, class_names=CLASS_NAMES)

        self.tracker.update(detections)

        # Build telemetry message
        msg_array = TrackedPlantArray() if TrackedPlantArray else None
        if msg_array:
            msg_array.header = color_msg.header

            for det in detections:
                rx, ry = det.root_point
                d_val = sample_depth_median(cv_depth, rx, ry, window_size=7, encoding="16UC1")
                det.root_d = d_val

                p_msg = TrackedPlant()
                p_msg.header = color_msg.header
                p_msg.track_id = int(det.track_id or -1)
                p_msg.class_id = det.class_id
                p_msg.class_name = det.class_name
                p_msg.confidence = float(det.confidence)
                p_msg.x1, p_msg.y1, p_msg.x2, p_msg.y2 = map(float, det.box_xyxy)
                p_msg.root_x, p_msg.root_y, p_msg.root_d = float(rx), float(ry), float(d_val)
                msg_array.plants.append(p_msg)

            self.pub_plants.publish(msg_array)

        # Publish visualization
        fps = 1.0 / max(1e-5, time.time() - t0)
        viz_img = draw_visualizations(cv_color, detections, fps=fps)
        viz_msg = self.bridge.cv2_to_imgmsg(viz_img, "bgr8")
        viz_msg.header = color_msg.header
        self.pub_viz.publish(viz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PlantPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
