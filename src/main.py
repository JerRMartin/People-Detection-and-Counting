# main.py
import config as C
from utils.helpers import show_frame, show_frame_by_path, preprocess_all_frames, show_all_frames, save_frame_with_detections
from utils.detection_and_counting import detect_people_in_frame_YOLO, detect_people_in_frame_FasterRCNN, detect_people_in_frame_HOG, do_ground_truth_comparison
import cv2

# Specify the frame number to process
frame_number = 250
RAW_FRAME_PATH = C.FRAME_DIRECTORY.RAW / f"seq_{int(frame_number):06d}.jpg"
PROCESSED_FRAME_PATH = C.FRAME_DIRECTORY.PROCESSED / f"processed_seq_{int(frame_number):06d}.jpg"

def do_detections_and_save_frames(frame_number):
    processed_frame_path = C.FRAME_DIRECTORY.PROCESSED / f"processed_seq_{int(frame_number):06d}.jpg"

    # YOLO Detection
    detection_frame, count = detect_people_in_frame_YOLO(processed_frame_path)
    # show_frame("YOLO", detection_frame)
    save_frame_with_detections(detection_frame, C.OUTPUT_DIRECTORY.YOLO / f"seq_{int(frame_number):06d}.jpg")
    print(f"[o] Detected people: {count} (YOLO)")

    # Faster R-CNN Detection
    detection_frame2, count2 = detect_people_in_frame_FasterRCNN(processed_frame_path)
    # show_frame("Faster R-CNN", detection_frame2)
    save_frame_with_detections(detection_frame2, C.OUTPUT_DIRECTORY.FASTER_RCNN / f"seq_{int(frame_number):06d}.jpg")
    print(f"[o] Detected people: {count2} (FASTER R-CNN)")

    # HOG Detection
    detection_frame3, count3 = detect_people_in_frame_HOG(processed_frame_path)
    # show_frame("HOG", detection_frame3)
    save_frame_with_detections(detection_frame3, C.OUTPUT_DIRECTORY.HOG / f"seq_{int(frame_number):06d}.jpg")
    print(f"[o] Detected people: {count3} (HOG)")


    # Wait for a key press to close all windows
    key = cv2.waitKey(0)  # Wait for a key press to show the next frame
    if key == ord('q'):  # Press 'q' to exit
        cv2.destroyAllWindows()

def compare_detections_to_ground_truth():
    for number in C.DETECTION_PRECISION_SUBSET:
        show_frame_by_path(C.OUTPUT_DIRECTORY.FASTER_RCNN / f"seq_{int(number):06d}.jpg")
        show_frame_by_path(C.OUTPUT_DIRECTORY.YOLO / f"seq_{int(number):06d}.jpg")
        show_frame_by_path(C.OUTPUT_DIRECTORY.HOG / f"seq_{int(number):06d}.jpg")
        show_frame_by_path(C.FRAME_DIRECTORY.GROUND_TRUTH / f"seq_{int(number):06d}_gt.png")

def main():
    # show_all_frames(C.FRAME_DIRECTORY.RAW)

    # Preprocess input frames
    # preprocess_all_frames()

    # Show original and processed version of frame 250
    # show_frame_by_path(RAW_FRAME_PATH)
    # show_frame_by_path(PROCESSED_FRAME_PATH)

    # Show detections
    # for number in C.DETECTION_PRECISION_SUBSET:
    #     do_detections_and_save_frames(number)

    # Ground Truth Comparison
    do_ground_truth_comparison(C.DETECTION_TYPE.YOLO)
    do_ground_truth_comparison(C.DETECTION_TYPE.FASTER_RCNN)
    do_ground_truth_comparison(C.DETECTION_TYPE.HOG)

    # compare_detections_to_ground_truth()
    

if __name__ == "__main__":
    main()
