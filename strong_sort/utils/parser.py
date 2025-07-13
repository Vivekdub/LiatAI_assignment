from yacs.config import CfgNode as CN

def get_config():
    _C = CN()

    # ---------------------- MODEL CONFIG ----------------------
    _C.MODEL = CN()
    _C.MODEL.REID_CKPT = "osnet_x0_25_msmt17.pt"  # good lightweight checkpoint (or upgrade to x0_5 or x1_0 for accuracy)

    # ---------------------- DETECTION CONFIG ----------------------
    _C.DETECTION = CN()
    _C.DETECTION.MIN_CONFIDENCE = 0.4  # Lower to keep more detections (useful in occlusion/low-res)
    _C.DETECTION.NMS_MAX_OVERLAP = 0.7  # More lenient NMS to keep multiple people close together

    # ---------------------- TRACKER CONFIG ----------------------
    _C.TRACKER = CN()
    _C.TRACKER.MAX_IOU_DISTANCE = 0.7     # IoU threshold for matching detections to tracks (lower = stricter)
    _C.TRACKER.MAX_AGE = 60               # Number of frames to keep "lost" tracks (shorter = better for fast recovery)
    _C.TRACKER.N_INIT = 3                 # Frames needed before confirming a track
    _C.TRACKER.NN_BUDGET = 100            # Number of features to store per track (lower = faster, higher = better identity memory)

    # ---------------------- DISTANCE METRIC CONFIG ----------------------
    _C.DISTANCE_METRIC = CN()
    _C.DISTANCE_METRIC.MAX_DIST = 0.4     # Appearance embedding threshold; < 0.5 is usually good for OSNet

    return _C
