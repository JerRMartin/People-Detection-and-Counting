# utils/detection_and_counting.py
from utils.helpers import get_new_color, save_frame_with_detections
import config as C
import cv2
import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import csv
import os

# ============= DETECTION ==============
# Load YOLO model
yolo_model = C.YOLO_MODEL_12X

# Load Faster R-CNN model
fasterrcnn_model = C.FASTER_RCNN_MODEL
fasterrcnn_model.eval()

def detect_people_in_frame_FasterRCNN(image_path):

    # Ensure output directory exists
    os.makedirs(C.OUTPUT_DIRECTORY.BASE, exist_ok=True)

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
                cv2.rectangle(frame, (int(x1), int(y1)), 
                              (int(x2), int(y2)), 
                              color, 2)
                cv2.putText(frame, f"person {people_count}", 
                            (int(x1), int(y1) - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 1)


    return frame, people_count

def detect_people_in_frame_YOLO(image_path):
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

    # Convert to x1,y1,x2,y2 format
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    colors = C.COLORS.copy()
    filtered = []

    # Filter out false positives based on box size
    MAX_W, MAX_H = 180, 340   # too big = false positive?
    # Draw detections
    for (x1, y1, x2, y2) in boxes:
        w = x2 - x1
        h = y2 - y1
        if w < MAX_W or h < MAX_H:
            filtered.append((x1, y1, x2, y2))

    for (x1, y1, x2, y2) in filtered:
        color, colors = get_new_color(colors)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    return frame, len(filtered)

def do_ground_truth_comparison(detection_type: C.DETECTION_TYPE):
    with open(f'{C.OUTPUT_DIRECTORY.BASE}/{detection_type}_output.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["frame_name", "true_count", "detected_count", "Absolute_Error"])
        writer.writeheader() 
        
        sum_of_AE = 0  
        for image_path, true_count in C.TRUE_PEOPLE_COUNT.items():
            match detection_type:
                case C.DETECTION_TYPE.HOG:
                    detection_frame, detected_count = detect_people_in_frame_HOG(f"processed_frames/processed_{image_path}")
                    save_frame_with_detections(detection_frame, C.OUTPUT_DIRECTORY.HOG / image_path) # write the image to a path
                    print(f"[o] Detected people: {detected_count} (HOG)")
                case C.DETECTION_TYPE.YOLO:
                    detection_frame, detected_count = detect_people_in_frame_YOLO(f"processed_frames/processed_{image_path}")
                    save_frame_with_detections(detection_frame, C.OUTPUT_DIRECTORY.YOLO / image_path) # write the image to a path
                    print(f"[o] Detected people: {detected_count} (YOLO)")
                case C.DETECTION_TYPE.FASTER_RCNN:
                    detection_frame, detected_count = detect_people_in_frame_FasterRCNN(f"processed_frames/processed_{image_path}")
                    save_frame_with_detections(detection_frame, C.OUTPUT_DIRECTORY.FASTER_RCNN / image_path) # write the image to a path
                    print(f"[o] Detected people: {detected_count} (FASTER R-CNN)")
                case _:
                    raise ValueError("Invalid detection type")

            abs_error = abs(true_count - detected_count)
            sum_of_AE += abs_error
            writer.writerow({
                "frame_name": image_path, 
                "true_count": true_count, 
                "detected_count": detected_count,
                "Absolute_Error": abs_error
            })

        # Write Mean Absolute Error row
        mean_absolute_error = sum_of_AE / len(C.TRUE_PEOPLE_COUNT)
        writer.writerow({
            "frame_name": "", 
            "true_count": "", 
            "detected_count": "Mean Absolute Error:",
            "Absolute_Error": mean_absolute_error
        })