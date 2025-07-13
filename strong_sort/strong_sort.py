import numpy as np
from deep_sort.deep_sort import DeepSort

class StrongSORT:
    def __init__(self, cfg, use_cuda=True):
        self.tracker = DeepSort(
            model_path=cfg.MODEL.REID_CKPT,
            max_dist=cfg.DISTANCE_METRIC.MAX_DIST,
            min_confidence=cfg.DETECTION.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DETECTION.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.TRACKER.MAX_IOU_DISTANCE,
            max_age=cfg.TRACKER.MAX_AGE,
            n_init=cfg.TRACKER.N_INIT,
            nn_budget=cfg.TRACKER.NN_BUDGET,
            use_cuda=use_cuda
        )

    def update(self, dets, frame):
        return self.tracker.update_tracks(dets, frame)