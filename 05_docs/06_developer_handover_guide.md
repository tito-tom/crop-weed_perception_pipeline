# Developer Onboarding & Project Handover Manual

> **Welcome to the Crop & Weed Perception Pipeline project!**  
> This guide is written specifically for new developers taking over this codebase. It provides a complete "Day 1 to Production" walkthrough, explains where every key component lives, and lists solutions to common troubleshooting errors.

---

## 🎯 Day 1: First-Time Setup Checklist

Follow these steps to get the full pipeline running on a new computer:

### 1. Environment & Dependencies
- **Python Environment**: Python 3.10+ recommended.
  ```bash
  git clone <REPO_URL>
  cd crop_weed_perception_pipeline
  pip install -r requirements.txt
  ```
- **Pretrained Weights**: Download `yolo11m-seg.pt` (or official Ultralytics weights) and place it inside `02_model_training/`.

### 2. Dataset Setup
- Place raw CVAT XML annotation files in `01_data_preparation/data/raw/xml/`.
- Place RGB source images in `01_data_preparation/data/raw/images/`.

---

## 🗺️ "Where Do I Make Changes?" Codebase Map

If you need to make modifications, refer to this reference table:

| Task / Feature Request | File to Edit | Key Function / Variable |
|---|---|---|
| Change learning rate, batch size, epochs | `02_model_training/config.py` | `LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS` |
| Add a 5th plant class (e.g., `fertilizer_pellet`) | `01_data_preparation/scripts/02_convert_xml_to_yolo.py` & `02_model_training/config.py` | `CLASS_MAPPING`, `NUM_CLASSES`, `CLASS_NAMES` |
| Adjust loss function gains (box, cls, root point) | `02_model_training/config.py` | `LOSS_BOX`, `LOSS_CLS`, `LOSS_KPT` |
| Change model network layers | `02_model_training/yolo_seg_root.yaml` | `head` architecture list |
| Change depth sampling window size | `03_deployment_ros2/plant_perception/config/params.yaml` | `depth_window_size` |
| Change ROS 2 camera input topics | `03_deployment_ros2/plant_perception/plant_perception/plant_perception_node.py` | `sub_color`, `sub_depth` topic names |
| Customize Web Dashboard UI | `04_web_dashboard/index.html` & `style.css` | HTML markup / CSS styles |

---

## ⚠️ Troubleshooting & Common Error Solutions

### 1. `FileNotFoundError: yolo11m-seg.pt`
* **Cause**: Pretrained base weights file is missing.
* **Fix**: Run `python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"` inside `02_model_training/` to auto-download standard weights.

### 2. ROS 2 Import Error: `No module named plant_perception.msg`
* **Cause**: Custom ROS 2 messages have not been built or sourced.
* **Fix**:
  ```bash
  cd 03_deployment_ros2/plant_perception
  colcon build --packages-select plant_perception
  source install/setup.bash
  ```

### 3. Depth Sampling Returns `0.0` Metres
* **Cause**: Depth camera image is not aligned with RGB frame or point is outside image boundaries.
* **Fix**: Verify your depth camera publisher is outputting aligned depth on `/camera/aligned_depth_to_color/image_raw` with encoding `16UC1` or `32FC1`.

### 4. Dashboard Shows "Disconnected" / WebSocket Error
* **Cause**: `rosbridge_server` is not running or port 9090 is blocked.
* **Fix**: Run `ros2 launch rosbridge_server rosbridge_websocket.launch.xml` on the host PC and check firewall settings (`sudo ufw allow 9090`).

---

## 💡 Key Design Decisions Explained

* **Why 7x7 Median Depth Window?**: Single-pixel depth readings are noisy due to specularity or holes in depth maps. Taking the median over a $7\times7$ patch centered on the stem root guarantees robust depth values.
* **Why Seed 42 for Dataset Splitting?**: Random splitting with seed `42` ensures every training experiment runs on identical train/val splits for fair model benchmarking.
* **Why Dual-Head Architecture?**: Predicting bounding box, polygon mask, and root keypoint in a single forward pass eliminates the overhead of running separate keypoint estimation models.
