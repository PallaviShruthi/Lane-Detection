from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = "runs/detect/lane_yolov8_final/weights/best.pt"
VIDEO_PATH = "input.mp4"

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)

print("Running debug — press Q to quit")
print("Shows raw YOLO boxes so we can see what it detects")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    result = model(cv2.resize(frame,(640,640)), conf=0.10, verbose=False)
    boxes  = result[0].boxes
    output = frame.copy()

    count = 0
    if boxes is not None:
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            x1=int(x1*w/640); y1=int(y1*h/640)
            x2=int(x2*w/640); y2=int(y2*h/640)
            conf = float(box.conf[0])
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            # Draw raw box
            cv2.rectangle(output,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(output,f"{conf:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            count += 1

    cv2.putText(output,f"Detections: {count}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    display = cv2.resize(output,(960,540))
    cv2.imshow("Debug", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()