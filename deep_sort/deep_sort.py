from deep_sort.deep.feature_extractor import Extractor
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
import numpy as np
import torch

class DeepSort:
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3,
                 nms_max_overlap=0.5, max_iou_distance=0.7, max_age=70,
                 n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_dist, nn_budget
        )
        self.tracker = Tracker(metric)
        self.use_cuda = use_cuda

    def update_tracks(self, detections, frame):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            return []

        bbox_xywh = []
        confs = []

        for *xyxy, conf in detections:
            x1, y1, x2, y2 = xyxy
            obj_w = x2 - x1
            obj_h = y2 - y1
            x_c = x1 + obj_w / 2
            y_c = y1 + obj_h / 2
            bbox_xywh.append([x_c, y_c, obj_w, obj_h])
            confs.append(conf)

        features = self.extractor(frame, bbox_xywh)
        detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(bbox_xywh, confs, features)]

        self.tracker.predict()
        self.tracker.update(detections)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, w, h = track.to_tlwh()
            x2, y2 = x1 + w, y1 + h
            outputs.append([x1, y1, x2, y2, track.track_id])

        return outputs
