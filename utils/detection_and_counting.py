# utils/detection_and_counting.py
import cv2
import numpy as np
import config as C
from utils.helpers import get_new_color

# TODO: Task 2: People Detection (25%) 
'''
Run detection on each frame, output bounding boxes for detected people, and filter obvious false positives (e.g., by size or region of interest).
'''
def detect_people_in_frame(frame) -> tuple[any, int]:
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

# TODO: Task 3: People Counting (15%)
'''
# Using your detection results, estimate the number of people present in each frame. Compare these estimated 
# counts with the ground-truth counts for the same frames and compute per-frame errors. Summarize
# your performance using appropriate metrics such as Mean Absolute Error (MAE), and discuss how well 
# your system counts people overall. If you incorporate temporal tracking to stabilize counts over time, 
# briefly explain your tracking approach and how it affects the counting accuracy.
'''
def compare_to_ground_truth(detections):
    # look at TRUE_PEOPLE_COUNT from config.py for ground truth, and compare with detections
    pass