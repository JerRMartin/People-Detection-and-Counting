# utils/helpers.py
import cv2
import config as C
from utils.processing import preprocess_frame
from random import randint

# Display a specific frame by its number and type
def show_frame_by_path(path):

    print(f"[o] Showing {path}")

    frame = cv2.imread(str(path))
    if frame is not None:
        
        # Resize frames
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        # Showing result
        cv2.imshow(f'{path}', frame)
        key = cv2.waitKey(0)  # Wait for a key press
        if key == ord('q'):  # Press 'q' to exit
            cv2.destroyAllWindows()

def show_frame(name, frame):
    cv2.imshow(name, frame)
    cv2.waitKey(0)

# Display all frames in the specified directory
def show_all_frames(dir: C.FRAME_DIRECTORY):

    print("[o] Press 'q' to quit. Press any other key to show the next frame.")

    for frame in dir.glob('*.jpg'):
        img = cv2.imread(str(frame))
        cv2.imshow(frame.name, img)
        key = cv2.waitKey(0)  # Wait for a key press to show the next frame
        if key == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()

# Preprocess all frames
def preprocess_all_frames(*kwargs):

    print("")
    print("[o] Preprocessing all frames in the frames directory...")

    for frame in C.FRAME_DIRECTORY.RAW.glob('*.jpg'):
        img = cv2.imread(str(frame))

        # Pre-Processing the Image
        preprocess_frame(frame.name, img, *kwargs)

    print("")
    print("[o] All frames were preprocessed.")


# Get a new color from the list of available colors, removing it from the list so it won't be reused until the list is exhausted
def get_new_color(colors: list) -> tuple[tuple[int, int, int], list]:
    if len(colors) == 0:
        colors = C.COLORS.copy()
    color = colors[randint(0,len(colors)-1)] 
    colors.remove(color)
    return (color, colors)


# Save frame with detections
def save_frame_with_detections(frame, output_path: C.OUTPUT_DIRECTORY):
    cv2.imwrite(str(output_path), frame)
    print(f"[o] Saved detection frame to: {output_path}")