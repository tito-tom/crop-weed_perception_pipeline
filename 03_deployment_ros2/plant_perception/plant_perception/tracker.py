#!/usr/bin/env python3
"""
tracker.py — BYTETracker Integration and Track ID Assignment
"""

import numpy as np

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    HAS_BYTETRACK = True
except Exception:
    BYTETracker = None
    HAS_BYTETRACK = False


class DetectionResults:
    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return DetectionResults(self.xywh[index], self.conf[index], self.cls[index])


class TrackerArgs:
    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1,
                 new_track_thresh=0.6, track_buffer=30, match_thresh=0.8):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return inter / float(areaA + areaB - inter)


class ByteTrackWrapper:
    ID_MAP_IOU_THRESH = 0.3

    def __init__(self):
        if HAS_BYTETRACK:
            self.tracker = BYTETracker(TrackerArgs())
        else:
            self.tracker = None

    def update(self, detections):
        if not HAS_BYTETRACK or self.tracker is None or not detections:
            return

        xywh, conf, cls = [], [], []
        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            w, h = x2 - x1, y2 - y1
            xywh.append([x1 + w / 2, y1 + h / 2, w, h])
            conf.append(det.confidence)
            cls.append(det.class_id)

        self.tracker.update(DetectionResults(xywh, conf, cls))

        active_tracks = [t for t in self.tracker.tracked_stracks if t.is_activated]
        pairs = []
        for d_idx, det in enumerate(detections):
            for t_idx, track in enumerate(active_tracks):
                iou = _iou(det.box_xyxy, track.xyxy)
                if iou >= self.ID_MAP_IOU_THRESH:
                    pairs.append((iou, d_idx, t_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)
        assigned_d, assigned_t = set(), set()
        for _, d_idx, t_idx in pairs:
            if d_idx not in assigned_d and t_idx not in assigned_t:
                detections[d_idx].track_id = int(active_tracks[t_idx].track_id)
                assigned_d.add(d_idx)
                assigned_t.add(t_idx)


class Tracker:
    def __init__(self, tracker_type="bytetrack"):
        self.wrapper = ByteTrackWrapper()

    def update(self, detections):
        self.wrapper.update(detections)
