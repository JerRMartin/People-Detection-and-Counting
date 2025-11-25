# main.py
import config as C
from utils.helpers import show_frame, show_all_frames, preprocess_all_frames, show_frame_by_number
from utils.detection_and_counting import detect_people_in_frame
import cv2

def main():
    # Preprocess input frames (your existing logic)
    preprocess_all_frames()

    # Show original and processed version of frame 250
    show_frame_by_number(250)
    show_frame_by_number(250, C.PROCESSED_FRAMES_DIR)

    # YOLO Detection
    frame_path = "processed_frames/processed_seq_000250.jpg"
    frame = cv2.imread(frame_path)
    detection_frame, count = detect_people_in_frame(frame)

    show_frame("YOLO Detections", detection_frame)
    print(f"[o] Detected people: {count}")

if __name__ == "__main__":
    main()
