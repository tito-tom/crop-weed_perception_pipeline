// app.js — Client-side web dashboard logic for ROS WebSocket connection

let ros = null;
let trackedPlantsTopic = null;
let compressedImageTopic = null;
let connected = false;
let ratioChart = null;

// UI Elements
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const ipInput = document.getElementById('rosbridge-ip');
const connectBtn = document.getElementById('connect-btn');
const logPanel = document.getElementById('log-panel');
const videoFeed = document.getElementById('video-feed');
const videoPlaceholder = document.getElementById('video-placeholder');
const feedFps = document.getElementById('feed-fps');

const cropCountEl = document.getElementById('crop-count');
const weedCountEl = document.getElementById('weed-count');
const totalCountEl = document.getElementById('total-count');
const detectionsTbody = document.getElementById('detections-tbody');

let frameCount = 0;
let lastFpsUpdate = Date.now();

function addLog(text, type = '') {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="log-time">[${time}]</span> ${text}`;
    logPanel.appendChild(entry);
    logPanel.scrollTop = logPanel.scrollHeight;
}

function initChart() {
    const ctx = document.getElementById('ratioChart').getContext('2d');
    ratioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Crops', borderColor: '#00ff87', backgroundColor: 'rgba(0, 255, 135, 0.1)', data: [], fill: true },
                { label: 'Weeds', borderColor: '#ff0055', backgroundColor: 'rgba(255, 0, 85, 0.1)', data: [], fill: true }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

function connectROS(ip) {
    const wsUrl = `ws://${ip}:9090`;
    addLog(`Attempting connection to ${wsUrl}...`);

    ros = new ROSLIB.Ros({ url: wsUrl });

    ros.on('connection', () => {
        connected = true;
        connectionDot.style.background = '#00ff87';
        connectionText.innerText = 'Connected';
        connectBtn.innerText = 'Disconnect';
        addLog('Connected to ROS 2 websocket server!', 'success');
        subscribeTopics();
    });

    ros.on('error', (error) => {
        addLog(`ROS Error: ${error}`, 'error');
    });

    ros.on('close', () => {
        connected = false;
        connectionDot.style.background = '#ff0055';
        connectionText.innerText = 'Disconnected';
        connectBtn.innerText = 'Connect';
        addLog('Disconnected from ROS 2 server.', 'error');
    });
}

function subscribeTopics() {
    trackedPlantsTopic = new ROSLIB.Topic({
        ros: ros,
        name: '/plant_perception/tracked_plants',
        messageType: 'plant_perception/msg/TrackedPlantArray'
    });

    trackedPlantsTopic.subscribe((message) => {
        updateTelemetry(message.plants);
    });

    compressedImageTopic = new ROSLIB.Topic({
        ros: ros,
        name: '/plant_perception/image_annotated/compressed',
        messageType: 'sensor_msgs/msg/CompressedImage'
    });

    compressedImageTopic.subscribe((message) => {
        videoFeed.src = 'data:image/jpeg;base64,' + message.data;
        videoFeed.style.display = 'block';
        videoPlaceholder.style.display = 'none';

        frameCount++;
        const now = Date.now();
        if (now - lastFpsUpdate >= 1000) {
            feedFps.innerText = `${frameCount} FPS`;
            frameCount = 0;
            lastFpsUpdate = now;
        }
    });
}

function updateTelemetry(plants) {
    let cropCount = 0;
    let weedCount = 0;
    detectionsTbody.innerHTML = '';

    plants.forEach(p => {
        if (p.class_name.includes('crop')) cropCount++;
        else weedCount++;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>#${p.track_id}</td>
            <td><span class="badge ${p.class_name.includes('crop') ? 'crop' : 'weed'}">${p.class_name}</span></td>
            <td>${(p.confidence * 100).toFixed(1)}%</td>
            <td>${p.root_d.toFixed(2)}m</td>
        `;
        detectionsTbody.appendChild(tr);
    });

    cropCountEl.innerText = cropCount;
    weedCountEl.innerText = weedCount;
    totalCountEl.innerText = plants.length;

    // Update Chart
    const timeLabel = new Date().toLocaleTimeString();
    ratioChart.data.labels.push(timeLabel);
    ratioChart.data.datasets[0].data.push(cropCount);
    ratioChart.data.datasets[1].data.push(weedCount);
    if (ratioChart.data.labels.length > 20) {
        ratioChart.data.labels.shift();
        ratioChart.data.datasets[0].data.shift();
        ratioChart.data.datasets[1].data.shift();
    }
    ratioChart.update();
}

connectBtn.addEventListener('click', () => {
    if (connected) {
        ros.close();
    } else {
        connectROS(ipInput.value);
    }
});

window.onload = () => {
    initChart();
};
