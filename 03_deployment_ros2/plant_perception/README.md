# Stage 3: Real-Time ROS 2 Deployment Package (`plant_perception`)

This module houses the ROS 2 package (`plant_perception`) for running real-time ONNX inference, object tracking, depth image sampling, and publishing plant telemetry onto a ROS 2 system (Humble / Jazzy).

---

## 📋 Package Structure

```
plant_perception/
├── CMakeLists.txt                      # ROS 2 build configuration
├── package.xml                         # Package dependencies manifest
├── msg/
│   ├── TrackedPlant.msg                # Single plant telemetry message
│   └── TrackedPlantArray.msg           # Array of tracked plants message
├── launch/
│   └── plant_perception.launch.py      # ROS 2 launch file
├── config/
│   └── params.yaml                     # Node parameter configuration
├── plant_perception/
│   ├── plant_perception_node.py        # Main perception ROS 2 node
│   ├── onnx_inferencer.py              # ONNX Runtime execution session
│   ├── tracker.py                      # BYTETracker multi-object tracking
│   ├── postprocess.py                  # NMS, mask decoding & coordinate scaling
│   ├── depth_utils.py                  # 7x7 median depth sampling at root point
│   └── visualization.py                # Bounding box, mask & root keypoint drawing
└── README.md
```

---

## ⚙️ Configuration (`config/params.yaml`)

Edit parameters in `config/params.yaml` to adjust settings:

```yaml
plant_perception_node:
  ros__parameters:
    model_path: "yolov11-seg-root.onnx"   # Path to ONNX model
    imgsz: 640                            # Model input resolution
    conf_thres: 0.25                      # Confidence threshold
    iou_thres: 0.45                       # NMS IoU threshold
    device: "cpu"                         # "cpu" or "cuda"
    depth_window_size: 7                  # Median depth window size (7x7 pixels)
```

---

## 💻 Step-by-Step Build & Execution Guide

### Step 1: Place Package in ROS 2 Workspace
Ensure `plant_perception` is placed inside your ROS 2 workspace `src/` folder (e.g. `ros2_ws/src/plant_perception`).

### Step 2: Build the ROS 2 Package
Source your ROS 2 installation (Humble / Jazzy) and build:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select plant_perception
source install/setup.bash
```

### Step 3: Run the Perception Node
Launch the node using the ROS 2 launch file:

```bash
ros2 launch plant_perception plant_perception.launch.py
```

Or run the executable node directly:

```bash
ros2 run plant_perception plant_perception_node.py --ros-args --params-file src/plant_perception/config/params.yaml
```

---

## 📡 ROS 2 Topics Reference

| Topic Name | Message Type | Direction | Description |
|---|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | Subscriber | Input RGB camera frame |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | Subscriber | Input 16-bit depth camera frame |
| `/plant_perception/tracked_plants` | `plant_perception/msg/TrackedPlantArray` | Publisher | Telemetry output (IDs, classes, box, root x/y/d) |
| `/plant_perception/image_annotated` | `sensor_msgs/msg/Image` | Publisher | Visual feed with drawn boxes, masks, and roots |

---

## 💡 Key Execution Flow for Maintainers

1. **Synchronized Callbacks**: `message_filters.ApproximateTimeSynchronizer` pairs RGB frames with depth frames within a 50ms window.
2. **ONNX Inference**: Runs preprocessed letterboxed frames through `onnxruntime.InferenceSession`.
3. **BYTETracker ID Assignment**: Keeps track of individual plants across consecutive frames so the robot arm can target specific IDs.
4. **Depth Sampling**: Computes depth $Z_{root}$ in metres by taking the median value of a $7 \times 7$ window around $(x_{root}, y_{root})$ to eliminate noise or missing pixels.
