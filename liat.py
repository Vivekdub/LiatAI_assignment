import cv2
import numpy as np
import sys
import torch
from ultralytics import YOLO
import os
# 👇️ Add correct path to StrongSORT repo
sys.path.append(os.path.abspath("."))
sys.path.append("strong_sort")

from strong_sort.strong_sort import StrongSORT
from strong_sort.utils.parser import get_config


# ----------------- Paths -----------------
VIDEO_PATH = '15sec_input_720p.mp4'
MODEL_PATH = 'best.pt'
OUTPUT_PATH = 'outputs/output_tracked.mp4'
CONFIG_PATH = 'C:/Users/Shashankd/OneDrive/Desktop/New folder/assingn_liatAI/StrongSORT/strong_sort/configs/strong_sort.yaml'

# ----------------- Load Model -----------------
model = YOLO("best.pt")

# ----------------- Init Tracker -----------------
cfg = get_config()
tracker = StrongSORT(cfg, use_cuda=True)
# cfg.merge_from_file("StrongSORT/application_util/configs/strong_sort.yaml")
# tracker = StrongSORT(cfg, use_cuda=torch.cuda.is_available())

# ----------------- Video I/O -----------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))
height = int(cap.get(4))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


def enhance_frame(frame):
    # 1. Denoise
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # # 2. Sharpen
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # frame = cv2.filter2D(frame, -1, kernel)

    # # 3. Histogram equalization
    # ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    # frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return frame

while True:
    ret, frame = cap.read()
    frame = enhance_frame(frame)
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        # detections.append([x1, y1, x2, y2, conf])
        if conf > 0.8: #and cls == 0:
            detections.append([x1, y1, x2, y2, conf])

    dets = np.array(detections) if detections else np.empty((0, 5))
    print(f"[DEBUG] Detections (before tracker): {dets}")
    print(f"[DEBUG] Detection shape: {dets.shape}")

    outputs = tracker.update(dets, frame)
    print(f"[DEBUG] Tracker outputs: {outputs}")


    for output in outputs:
        x1, y1, x2, y2, track_id = output[:5]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out.write(frame)
    # cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == "__main__":
    cap.release()
    out.release()
    cv2.destroyAllWindows()

