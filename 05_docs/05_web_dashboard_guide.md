# Stage 4 Documentation: Web Dashboard & Monitoring Guide

## 1. Web Architecture

The monitoring dashboard provides real-time telemetry visualization over WebSockets:

- **rosbridge_server**: ROS 2 WebSocket server bridge on port `9090`.
- **roslibjs**: Client-side JavaScript library subscribing to `/plant_perception/tracked_plants` and `/plant_perception/image_annotated/compressed`.
- **Chart.js**: Dynamic line graph rendering crop vs. weed tracking trends over time.

## 2. Dashboard Interface Features

1. **Live Camera Feed**: Displays live camera overlay with bounding boxes, keypoint markers, and instance masks.
2. **Telemetry Indicators**: Displays real-time counts for active crops, weeds, and total detections.
3. **Active Target Table**: Tabulates active track IDs, class tags, confidence scores, and root depth estimates.
