"""
STEP 2 — Run after convert_labels.py
Trains YOLOv8 detection model on your lane dataset
Estimated time: 1.5 hours on CPU
"""
from ultralytics import YOLO

print("=" * 50)
print("  YOLOv8 Lane Detection Training")
print("  Dataset: C:/rf_data")
print("  Estimated time: 1.5 hrs on CPU")
print("=" * 50)

model = YOLO("yolov8s.pt")  # detection model (not seg)

results = model.train(
    data    = "C:/rf_data/data.yaml",
    epochs  = 30,
    imgsz   = 416,
    batch   = 4,
    name    = "lane_yolov8_det",
    patience= 10,
    device  = "cpu",
    workers = 2,

    # Augmentation tuned for your roads:
    # tree shadows, dashed white lines, green/yellow kerbs
    hsv_v   = 0.5,   # brightness variation → handles shadows
    hsv_s   = 0.4,   # saturation → handles dusty/faded markings
    hsv_h   = 0.02,  # hue shift
    fliplr  = 0.5,   # horizontal flip
    degrees = 5.0,   # slight rotation
    scale   = 0.4,   # zoom variation
    mosaic  = 1.0,   # combines 4 images → great for small dataset

    verbose = True
)

print("\n✅ Training Complete!")
print("📁 Model: runs/detect/lane_yolov8_det/weights/best.pt")
