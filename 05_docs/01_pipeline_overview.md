# Comprehensive System Architecture & Pipeline Overview

## Executive Summary

This document presents an end-to-end computer vision and robotics perception pipeline designed for automated agricultural weeding systems. The system integrates instance segmentation, plant root keypoint localization, real-time object tracking, depth estimation, and web telemetry into a unified architecture.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             END-TO-END PIPELINE WORKFLOW                         │
└──────────────────────────────────────────────────────────────────────────────────┘

 [Stage 1: Annotation Processing]
    CVAT XML Annotations (Polygons + Keypoint Root Tags)
                │
                ▼
    02_convert_xml_to_yolo.py ──► Standard 4-Class YOLO Format

 [Stage 2: Dual-Head Deep Learning Model]
    YOLOv11 Medium Backbone + FPN/PAN Neck
                │
                ├──► Bounding Box Branch (CIoU Loss)
                ├──► Instance Segmentation Branch (BCE Prototype Loss)
                ├──► Classification Branch (BCE Loss)
                └──► Root Keypoint Regression Branch (L1 Loss)  ⭐
                │
                ▼
    export_onnx.py ──► Lightweight ONNX Runtime Model

 [Stage 3: ROS 2 Deployment Node]
    RGB-D Camera Input (/camera/color, /camera/depth)
                │
                ▼
    plant_perception_node.py
                ├──► ONNX Inference Session
                ├──► BYTETracker ID Assignment
                ├──► Median Depth Sampling at Root Coordinates
                └──► Telemetry Publisher (/plant_perception/tracked_plants)

 [Stage 4: Web Dashboard & Visual Monitoring]
    rosbridge_server (WebSocket Protocol)
                │
                ▼
    Browser UI (Express / Chart.js / Socket.IO)
```

## System Requirements & Key Innovations

1. **Dual-Head Multi-Task Learning**: Predicts bounding box, class, polygon mask, and 2D root coordinate $(x_{root}, y_{root})$ in a single forward pass.
2. **Precision Root Point Localization**: Direct root point prediction enables accurate robotic end-effector targeting for weed destruction without relying on bounding box center heuristics.
3. **RGB-D Spatial Fusion**: Couples 2D keypoints with aligned 16-bit depth sensor readings to produce 3D root coordinates $(x, y, z)$.
4. **Real-Time Performance**: Optimized with ONNX Runtime to achieve low-latency processing suitable for field deployment on embedded hardware (NVIDIA Jetson / x86 PCs).
