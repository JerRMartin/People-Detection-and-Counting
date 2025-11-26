# main.py
import config as C
from utils.helpers import show_frame, show_frame_by_number, preprocess_all_frames, show_all_frames
from utils.detection_and_counting import detect_people_in_frame_YOLO, detect_people_in_frame_FasterRCNN, detect_people_in_frame_HOG
import cv2

# Specify the frame number to process
frame_number = 250
RAW_FRAME_PATH = C.FRAME_DIRECTORY.RAW / f"seq_{int(frame_number):06d}.jpg"
PROCESSED_FRAME_PATH = C.FRAME_DIRECTORY.PROCESSED / f"processed_seq_{int(frame_number):06d}.jpg"

def main():
    # show_all_frames(C.FRAME_DIRECTORY.RAW)

    # Preprocess input frames
    # preprocess_all_frames()

    # Show original and processed version of frame 250
    show_frame_by_number(RAW_FRAME_PATH)
    show_frame_by_number(PROCESSED_FRAME_PATH)

    # YOLO Detection
    detection_frame, count = detect_people_in_frame_YOLO(PROCESSED_FRAME_PATH)
    show_frame("YOLO", detection_frame)
    print(f"[o] Detected people: {count} (YOLO)")

    # Faster R-CNN Detection
    detection_frame2, count2 = detect_people_in_frame_FasterRCNN(PROCESSED_FRAME_PATH)
    show_frame("Faster R-CNN", detection_frame2)
    print(f"[o] Detected people: {count2} (FASTER R-CNN)")

    # HOG Detection
    detection_frame3, count3 = detect_people_in_frame_HOG(PROCESSED_FRAME_PATH)
    show_frame("HOG", detection_frame3)
    print(f"[o] Detected people: {count3} (HOG)")


    # Wait for a key press to close all windows
    key = cv2.waitKey(0)  # Wait for a key press to show the next frame
    if key == ord('q'):  # Press 'q' to exit
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
