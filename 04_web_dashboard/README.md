# Stage 4: Real-Time Web Telemetry Dashboard

This directory contains the real-time web monitoring application. It bridges ROS 2 topics to a browser user interface over WebSocket protocols using `rosbridge_server`, `roslibjs`, and `Chart.js`.

---

## 📋 Directory Structure

```
04_web_dashboard/
├── package.json        # Node.js dependencies (Express)
├── app.js              # Client-side JavaScript WebSocket logic
├── index.html          # Main HTML5 dashboard UI
├── style.css           # Custom dark-mode design system
└── README.md
```

---

## 💻 Step-by-Step Usage Guide

### Step 1: Install Dependencies
Install Node.js packages (if serving via Express server):

```bash
cd 04_web_dashboard
npm install
```

### Step 2: Launch ROS 2 WebSocket Server (`rosbridge_server`)
On your ROS 2 host or robot PC (e.g. Jetson Orin), launch `rosbridge_server` to open WebSocket port `9090`:

```bash
# Install rosbridge_server if not already present
sudo apt install ros-humble-rosbridge-server   # for ROS 2 Humble
# or: sudo apt install ros-jazzy-rosbridge-server

# Launch WebSocket Bridge
ros2 launch rosbridge_server rosbridge_websocket.launch.xml
```

### Step 3: Launch the Dashboard Server
Start the local Express server:

```bash
npm start
```

Open your browser and navigate to:
```
http://localhost:3000
```
*(Alternatively, you can open `index.html` directly in any web browser without running Node.js).*

### Step 4: Connect to Robot Telemetry
1. Enter the IP address of your ROS 2 device (e.g. `127.0.0.1` for local testing, or `192.168.x.x` for Jetson).
2. Click **Connect**.
3. View real-time live camera stream, plant counts (Crops vs Weeds), dynamic Chart.js tracking graph, and active target depth measurements.

---

## 💡 Troubleshooting & Connection Tips

* **WebSocket Port Error**: Ensure port `9090` is open on your firewall if connecting over a local network (`sudo ufw allow 9090`).
* **No Video Feed**: Ensure `plant_perception_node.py` is running and publishing to `/plant_perception/image_annotated`.
