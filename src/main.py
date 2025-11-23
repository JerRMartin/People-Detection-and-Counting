# main.py
import config as C
from utils.helpers import show_frame, show_all_frames, preprocess_all_frames, show_frame_by_number
from utils.detection_and_counting import detect_people_in_frame
import cv2

def main():
    #show_all_frames()
    preprocess_all_frames()

    show_frame_by_number(250)
    show_frame_by_number(250, C.PROCESSED_FRAMES_DIR)
    frame, count = detect_people_in_frame(cv2.imread("processed_frames/processed_seq_000250.jpg"))
    show_frame("Detections", frame)
    print(f"[o] Detected people: {count}")

if __name__ == "__main__":
    main()

