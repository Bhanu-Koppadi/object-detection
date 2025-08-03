
import cv2
import torch
from ultralytics import YOLO

# STEP 1: Select Model
USE_MODEL = "yolov8"  # choose "yolov5" or "yolov8"

if USE_MODEL == "yolov5":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
else:
    model = YOLO("yolov8l.pt")

print(f"[INFO] {USE_MODEL} model loaded successfully!")

# STEP 2: Initialize Webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    print("❌ ERROR: Could not access webcam.")
    exit()

print("[INFO] Webcam started. Press 'q' or ESC to quit.")

# STEP 3: Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Perform detection
    results = model(frame, conf=0.5, imgsz=640)
    annotated_frame = results[0].plot()

    # ✅ Print detected objects
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"Detected: {label} ({conf:.2f})")

    cv2.imshow("YOLO Live Detection", annotated_frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27 or cv2.getWindowProperty("YOLO Live Detection", cv2.WND_PROP_VISIBLE) < 1:
        print("[INFO] Exiting live detection...")
        break

cap.release()
cv2.destroyAllWindows()