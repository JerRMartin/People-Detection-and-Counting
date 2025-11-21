# utils/helpers.py
import cv2
import config as C
from utils.processing import preprocess_frame

# Retrieve a specific frame by its number
def get_frame(number) -> tuple[str,  cv2.typing.MatLike | None] | None:
    filename = f"seq_{int(number):06d}.jpg"
    path = C.FRAMES_DIR / filename
    if path.exists():
        img = cv2.imread(str(path))
        return (filename, img)
    return None

# Display a specific frame by its number
def show_frame(number):

    print("[o] Showing frame number " + str(number))

    frame_data = get_frame(number)
    if frame_data:
        name, img = frame_data

        # Pre-Processing the Image
        preprocessed = preprocess_frame(name, img)
        name = preprocessed[0]
        img = preprocessed[1]

        # Showing result
        cv2.imshow(name, img)
        key = cv2.waitKey(0)  # Wait for a key press
        if key == ord('q'):  # Press 'q' to exit
            cv2.destroyAllWindows()

# Display all frames in the frames directory
def show_all_frames():

    print("[o] Press 'q' to quit. Press any other key to show the next frame.")

    for frame in C.FRAMES_DIR.glob('*.jpg'):
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

    for frame in C.FRAMES_DIR.glob('*.jpg'):
        img = cv2.imread(str(frame))

        # Pre-Processing the Image
        preprocess_frame(frame.name, img)

    print(".")
    print("[o] All frames were preprocessed.")

    return 




# Pre-Processing the Image
#preprocessed = preprocess_frame(frame.name, img)
#name = preprocessed[0]
#img = preprocessed[1]