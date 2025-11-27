# utils/detection_and_counting.py
from utils.helpers import get_new_color
import config as C
import cv2
import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import csv


# ============= DETECTION ==============

# Load YOLO model
yolo_model = C.YOLO_MODEL_12X

# Load Faster R-CNN model
fasterrcnn_model = C.FASTER_RCNN_MODEL
fasterrcnn_model.eval()

def detect_people_in_frame_FasterRCNN(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    colors = C.COLORS.copy()
    frame = cv2.imread(str(image_path))

    with torch.no_grad():
        outputs = fasterrcnn_model([image_tensor])

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    people_count = 0

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.8:
            if label.item() == 1: # class 1 = person in COCO dataset https://cocodataset.org/#home
                color, colors = get_new_color(colors)
                x1, y1, x2, y2 = box
                people_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"person {people_count}", (int(x1), int(y1) - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    return frame, people_count

def detect_people_in_frame_YOLO(image_path):
    """
    todo: Task 2: People Detection (25%) 
    Run detection on each frame, output bounding boxes for detected people, and filter obvious
    false positives (e.g., by size or region of interest).
    """
    frame = cv2.imread(str(image_path))
    results = yolo_model(frame)[0]

    people_count = 0
    colors = C.COLORS.copy()

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # class 0 = person 
            color, colors = get_new_color(colors)
            people_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"person {people_count}", (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame, people_count


def detect_people_in_frame_HOG(image_path) -> tuple[any, int]:
    frame = cv2.imread(str(image_path))
    # Initialize the HOG + SVM detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Run the detector
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.05
    )

    # Filter out false positives based on box size
    MAX_W, MAX_H = 180, 340   # too big = false

    # Convert to x1,y1,x2,y2 format
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    colors = C.COLORS.copy()
    filtered = []

    # Draw detections
    for (x1, y1, x2, y2) in boxes:
        w = x2 - x1
        h = y2 - y1
        if w > MAX_W or h > MAX_H:
            continue
        filtered.append((x1, y1, x2, y2))

    for (x1, y1, x2, y2) in filtered:
        color, colors = get_new_color(colors)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    return frame, len(filtered)


# ============= COUNTING ==============
#  Task 3: People Counting (15%)
'''
# Using your detection results, estimate the number of people present in each frame. Compare these 
# estimated counts with the ground-truth counts for the same frames and compute per-frame errors. 
# Summarize your performance using appropriate metrics such as Mean Absolute Error (MAE), and discuss
# how well your system counts people overall. If you incorporate temporal tracking to stabilize counts
# over time, briefly explain your tracking approach and how it affects the counting accuracy.
''' 

def do_ground_truth_comparison(detection_type: C.DETECTION_TYPE):
    with open(f'{detection_type}_output.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["frame_name", "true_count", "detected_count", "Mean_Error"])
        writer.writeheader()  # Write header once

        sum_of_ME = 0
        
        for image_path, true_count in C.TRUE_PEOPLE_COUNT.items():
            match detection_type:
                case C.DETECTION_TYPE.HOG:
                    _, detected_count = detect_people_in_frame_HOG(image_path)
                case C.DETECTION_TYPE.YOLO:
                    _, detected_count = detect_people_in_frame_YOLO(image_path)
                case C.DETECTION_TYPE.FASTER_RCNN:
                    _, detected_count = detect_people_in_frame_FasterRCNN(image_path)
                case _:
                    raise ValueError("Invalid detection type")
                
            mean_error = abs(true_count - detected_count)
            sum_of_ME += mean_error
            data = {
                "frame_name": image_path, 
                "true_count": true_count, 
                "detected_count": detected_count,
                "Mean_Error": mean_error
            }
            writer.writerow(data)  # Append each row

        writer.writerow({
            "frame_name": "", 
            "true_count": "", 
            "detected_count": "Mean Absolute Error:",
            "Mean_Error": sum_of_ME/len(C.TRUE_PEOPLE_COUNT)
            }
        )

