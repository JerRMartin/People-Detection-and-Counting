# main.py
import argparse
import sys
from pathlib import Path
import config as C
from utils.helpers import show_frame_by_path, preprocess_all_frames, show_all_frames, save_frame_with_detections
from utils.detection_and_counting import detect_people_in_frame_YOLO, detect_people_in_frame_FasterRCNN, detect_people_in_frame_HOG, do_ground_truth_comparison
import cv2
import os

def preprocess_frames():
    """Preprocess all frames using the specified image adjustments."""
    preprocess_all_frames(C.IMAGE_ADJUSTMENT.SHARPEN, C.IMAGE_ADJUSTMENT.CONTRAST_ENHANCEMENT)
    print("Frame preprocessing completed.")

def show_specific_frame(frame_number: int):
    """Show the raw and processed version of a specific frame."""
    raw_file_name = f"seq_{int(frame_number):06d}.jpg"
    RAW_FRAME_PATH = C.FRAME_DIRECTORY.RAW / raw_file_name
    PROCESSED_FRAME_PATH = C.FRAME_DIRECTORY.PROCESSED / f"processed_seq_{int(frame_number):06d}.jpg"
    
    if not RAW_FRAME_PATH.exists():
        print(f"Error: Raw frame {raw_file_name} does not exist.")
        return
    
    print(f"Displaying frame {frame_number}:")
    show_frame_by_path(RAW_FRAME_PATH)
    show_frame_by_path(PROCESSED_FRAME_PATH)

def show_all_frames_func():
    """Display all frames in the raw directory."""
    show_all_frames(C.FRAME_DIRECTORY.RAW)

def run_ground_truth_comparison(detection_type):

    # Ensure output directory exists
    os.makedirs(C.OUTPUT_DIRECTORY.BASE, exist_ok=True)

    """Run ground truth comparison for a specific detection method."""
    valid_types = [C.DETECTION_TYPE.YOLO, C.DETECTION_TYPE.FASTER_RCNN, C.DETECTION_TYPE.HOG]
    
    if detection_type not in valid_types:
        print(f"Error: Invalid detection type. Valid types are: {[t.name for t in valid_types]}")
        return
    
    print(f"Running ground truth comparison for {detection_type}...")
    do_ground_truth_comparison(detection_type)
    print(f"Ground truth comparison completed for {detection_type}.")

def run_full_pipeline():
    """Run the complete processing pipeline."""
    print("Starting full processing pipeline...")
    preprocess_frames()
    
    # Show example frame
    frame_number = 250
    show_specific_frame(frame_number)
    
    # Run ground truth comparisons for all detection methods
    for detection_type in [C.DETECTION_TYPE.YOLO, C.DETECTION_TYPE.FASTER_RCNN, C.DETECTION_TYPE.HOG]:
        run_ground_truth_comparison(detection_type)
    
    print("Full processing pipeline completed.")

def main():
    parser = argparse.ArgumentParser(description="Person detection and frame processing toolkit")
    
    parser.add_argument(
        'command',
        choices=['preprocess', 'show-frame', 'show-all-frames', 'ground-truth', 'full-pipeline'],
        help="Command to execute"
    )
    
    parser.add_argument(
        '--frame-number',
        type=int,
        help="Frame number to process (required for show-frame command)"
    )
    
    parser.add_argument(
        '--detection-type',
        choices=['YOLO', 'FASTER_RCNN', 'HOG'],
        help="Detection type for ground-truth comparison"
    )
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        preprocess_frames()
    
    elif args.command == 'show-frame':
        if args.frame_number is None:
            parser.error("Frame number is required for show-frame command. Use --frame-number <number>")
        show_specific_frame(args.frame_number)
    
    elif args.command == 'show-all-frames':
        show_all_frames_func()
    
    elif args.command == 'ground-truth':
        if args.detection_type is None:
            parser.error("Detection type is required for ground-truth command. Use --detection-type <type>")
        
        detection_map = {
            'YOLO': C.DETECTION_TYPE.YOLO,
            'FASTER_RCNN': C.DETECTION_TYPE.FASTER_RCNN,
            'HOG': C.DETECTION_TYPE.HOG
        }
        detection_type = detection_map.get(args.detection_type)
        run_ground_truth_comparison(detection_type)
    
    elif args.command == 'full-pipeline':
        run_full_pipeline()

if __name__ == "__main__":
    main()