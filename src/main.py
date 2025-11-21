# main.py
import cv2
from pathlib import Path
from utils.helpers import show_frame, show_all_frames, preprocess_all_frames

# Loading Face Detection models
prototxt = r"C:\opencv_models\deploy.prototxt"
weights  = r"C:\opencv_models\res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))

def detect_faces_in_frames():
    frames_dir = Path("processed_frames")

    for frame_path in frames_dir.glob("*.jpg"):
        img = cv2.imread(str(frame_path))
        h, w = img.shape[:2]

        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        face_net.setInput(blob)
        detections = face_net.forward()

        # Draw boxes
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.imshow("Faces", img)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

def main():
    #show_frame(250)
    #show_all_frames()
    #preprocess_all_frames()

    detect_faces_in_frames()



if __name__ == "__main__":
    main()

