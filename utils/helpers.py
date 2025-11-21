# utils/helpers.py
import cv2
import config as C

def get_frame(number) -> tuple[str,  cv2.typing.MatLike | None] | None: 
    filename = f"seq_{int(number):06d}.jpg"
    path = C.FRAMES_DIR / filename
    if path.exists():
        img = cv2.imread(str(path))
        return (filename, img)
    return None

def show_frame(number):

    print("[o] Showing frame number " + str(number))

    frame_data = get_frame(number)
    if frame_data:
        name, img = frame_data
        cv2.imshow(name, img)
        key = cv2.waitKey(0)  # Wait for a key press to close the frame
        if key == ord('q'):  # Press 'q' to exit
            cv2.destroyAllWindows()

def show_all_frames():

    print("[o] Press 'q' to quit. Press any other key to show the next frame.")

    for frame in C.FRAMES_DIR.glob('*.jpg'):
        img = cv2.imread(str(frame))
        cv2.imshow("Frame", img)
        key = cv2.waitKey(0)  # Wait for a key press to show the next frame
        if key == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()