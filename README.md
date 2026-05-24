# 🚗 Lane Detection using YOLOv8

A real-time lane detection system built with **YOLOv8**, trained on custom road footage. Detects lane markings from video input using a fine-tuned object detection model.

---

## 📁 Project Structure

```
lane_detection/
│
├── download_dataset.py     # Downloads dataset from Roboflow
├── extract_frames.py       # Extracts frames from local videos
├── check_dataset.py        # Validates dataset before training
├── Convert_label.py        # Converts segmentation labels → YOLO bounding boxes
├── train.py                # Trains YOLOv8 model on lane dataset
├── debug_detect.py         # Debugs detection with raw YOLO boxes
├── live_lane.py            # Real-time lane detection on video
├── input.mp4               # Sample input video
├── best.pt                 # Best trained model weights
└── last.pt                 # Last checkpoint model weights
```

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- Roboflow

Install dependencies:

```bash
pip install ultralytics opencv-python roboflow pyyaml
```

---

## 🚀 How to Run

### Step 1 — Download Dataset
```bash
python download_dataset.py
```
Downloads the lane detection dataset from Roboflow into your local directory.

### Step 2 — (Optional) Extract Frames from Your Own Videos
```bash
python extract_frames.py
```
Extracts 1 frame every 20 frames from videos in `C:/Users/DELL/Downloads/my_videos` and saves them to `my_road_frames/`.

### Step 3 — Check Dataset
```bash
python check_dataset.py
```
Validates that images, labels, and `data.yaml` are correct before training.

### Step 4 — Convert Labels
```bash
python Convert_label.py
```
Converts segmentation polygon labels to YOLOv8 detection bounding box format.

### Step 5 — Train the Model
```bash
python train.py
```
Trains a YOLOv8s model for 30 epochs on the lane dataset.
> ⏱️ Estimated time: ~1.5 hours on CPU

Trained weights will be saved to:
```
runs/detect/lane_yolov8_det/weights/best.pt
```

### Step 6 — Debug Detection
```bash
python debug_detect.py
```
Runs detection on `input.mp4` and shows raw bounding boxes. Press `Q` to quit.

### Step 7 — Run Live Lane Detection
```bash
python live_lane.py
```
Runs real-time lane detection on the input video using the trained model.

---

## 🧠 Model Details

| Property        | Value                  |
|----------------|------------------------|
| Architecture    | YOLOv8s (detection)    |
| Input Size      | 416 × 416              |
| Epochs          | 30                     |
| Batch Size      | 4                      |
| Device          | CPU                    |
| Classes         | 1 (`lane`)             |
| Dataset Source  | Roboflow               |

### Augmentations Used
| Augmentation | Value | Purpose |
|---|---|---|
| `hsv_v` | 0.5 | Handles shadows from trees |
| `hsv_s` | 0.4 | Handles dusty/faded markings |
| `fliplr` | 0.5 | Horizontal flip |
| `degrees` | 5.0 | Slight rotation |
| `mosaic` | 1.0 | Good for small datasets |

---

## 📊 Dataset

- **Source:** [Roboflow — lane_train_v6](https://roboflow.com)
- **Workspace:** `lanedetection-lihli`
- **Format:** YOLOv8 detection
- **Classes:** `lane`

---

## 📸 Sample Output

> Real-time lane boundaries detected.

> <img width="1421" height="731" alt="Screenshot (43)" src="https://github.com/user-attachments/assets/0c4fa957-7008-4327-9019-96421964a153" />



---

## 📝 Notes

- Model was trained on Indian road conditions with tree shadows, dashed white lines, and green/yellow kerbs.
- Use `best.pt` for inference (highest validation accuracy).
- Use `last.pt` only if you want to resume training.

---

## 👩‍💻 Author

**Pallavi Shruthi**
[GitHub](https://github.com/PallaviShruthi)
