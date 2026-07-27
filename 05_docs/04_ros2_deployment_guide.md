# Stage 3 Documentation: ROS 2 Deployment & Depth Integration

## 1. ROS 2 Node Architecture

The `plant_perception_node` operates as a real-time ROS 2 processing pipeline:

1. **Approximate Time Synchronization**: Synchronizes RGB frames with 16-bit aligned depth frames.
2. **ONNX Forward Execution**: Evaluates letterboxed frame through ONNX Runtime engine.
3. **NMS & Keypoint Rescaling**: Applies Non-Maximum Suppression and maps keypoints back to native camera resolution.
4. **BYTETracker ID Tracking**: Associates detection bounding boxes across consecutive frames using Kalman filtering and IoU matching.
5. **Median Depth Sampling**: Samples depth in a $7 \times 7$ window around root point coordinate $(x_{root}, y_{root})$:

$$ Z_{root} = \text{median} \left( \{ D(x, y) \mid (x, y) \in W_{7\times7}(x_{root}, y_{root}), D(x,y) > 0 \} \right) $$

## 2. Message Definition Reference

### `TrackedPlant.msg`
```
std_msgs/Header header
int32 track_id
int32 class_id
string class_name
float32 confidence
float32 x1, y1, x2, y2
float32 root_x, root_y, root_d
sensor_msgs/Image mask
```
