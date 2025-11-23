# utils/helpers.py
import cv2
import config as C
from utils.processing import preprocess_frame
from random import randint

# Retrieve a specific frame by its number and type
def _get_frame(number, type:C.FRAME_TYPE) -> tuple[str,  cv2.typing.MatLike | None] | None:
    prefix = "processed_seq" if type == C.PROCESSED_FRAMES_DIR else "seq"
    filename = f"{prefix}_{int(number):06d}.jpg"
    path = type / filename
    if path.exists():
        img = cv2.imread(str(path))
        return (filename, img)
    return None

# Display a specific frame by its number and type
def show_frame_by_number(number, type:C.FRAME_TYPE = C.RAW_FRAMES_DIR):

    print(f"[o] Showing {type} number {number}")

    frame_data = _get_frame(number, type)
    if frame_data:
        name, img = frame_data
        
        # Showing result
        cv2.imshow(f'{name} ({type})', img)
        key = cv2.waitKey(0)  # Wait for a key press
        if key == ord('q'):  # Press 'q' to exit
            cv2.destroyAllWindows()

def show_frame(name, frame):
    cv2.imshow(name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display all frames in the specified directory
def show_all_frames(type:C.FRAME_TYPE = C.RAW_FRAMES_DIR):

    print("[o] Press 'q' to quit. Press any other key to show the next frame.")

    for frame in type.glob('*.jpg'):
        img = cv2.imread(str(frame))
        cv2.imshow(img)
        key = cv2.waitKey(0)  # Wait for a key press to show the next frame
        if key == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()

# Preprocess all frames
def preprocess_all_frames():

    print(".")
    print("[o] Preprocessing all frames in the frames directory...")

    for frame in C.RAW_FRAMES_DIR.glob('*.jpg'):
        img = cv2.imread(str(frame))

        # Pre-Processing the Image
        preprocess_frame(frame.name, img)

    print(".")
    print("[o] All frames were preprocessed.")

    return 

# Get a new color from the list of available colors, removing it from the list so it won't be reused until the list is exhausted
def get_new_color(colors: list) -> tuple[tuple[int, int, int], list]:
    if len(colors) == 0:
        colors = C.COLORS.copy()
    color = colors[randint(0,len(colors)-1)] 
    colors.remove(color)
    return (color, colors)


# Pre-Processing the Image
#preprocessed = preprocess_frame(frame.name, img)
#name = preprocessed[0]
#img = preprocessed[1]