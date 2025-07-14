
# Re-Identification in a Single Feed – LIAT AI Assignment

This project addresses the problem of real-time re-identification of soccer players in a single 15-second video using object detection and tracking techniques.

---

## 🎯 Objective

Given a short video (`15sec_input_720p.mp4`), detect and assign unique IDs to each player using YOLOv11. Ensure that if a player exits the frame and reappears, the same ID is reassigned using StrongSORT tracking logic.

---

## 🧠 System Overview

- **Object Detection**: YOLOv11 (fine-tuned) using `best.pt`
- **Tracking**: StrongSORT with Deep ReID + Kalman Filter
- **Real-Time Simulation**: Frame-by-frame tracking with ID consistency

---

## 📂 Directory Structure

```
LiatAI_assignment/
├── deep_sort/                 # DeepSORT module
├── strong_sort/               # StrongSORT module with Deep ReID
├── outputs/                   # Tracked output video is saved here
├── best.pt                    # Fine-tuned YOLOv11 model
├── 15sec_input_720p.mp4       # Provided input video
├── liat.py                    # Main pipeline script
└── test.py                    # Optional testing script
```

---

## ▶️ Running the Project

> ⚠️ Ensure Python 3.8+ is installed with necessary dependencies.


```bash
pip install -r requirments.txt
```

Then, run the script:

```bash
python liat.py
```

Output video will be saved at:

```
outputs/output_tracked.mp4
```

---

## 🔧 Configuration

Make sure to update this path in your script based on your system:

```python
CONFIG_PATH = "strong_sort/configs/strong_sort.yaml"
```

---

## ✅ Features

- Robust Re-Identification with appearance + motion tracking
- Real-time simulation via sequential frame processing
- Optional frame enhancement and noise reduction
- Automatically reassigns same IDs to players re-entering the scene

---

## 📝 Notes

- The tracker requires a CUDA-compatible GPU for best performance.
- `best.pt` is a YOLOv11 model pre-trained to detect players and the ball.

---

## 🙌 Acknowledgements

- YOLOv11 by [Ultralytics](https://github.com/ultralytics/yolov5)
- StrongSORT adapted with Deep ReID embeddings
