from pathlib import Path
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataclasses import dataclass

@dataclass
class FRAME_DIRECTORY:
    RAW: Path = Path("frames")
    PROCESSED: Path = Path("processed_frames")
    GROUND_TRUTH: Path = Path("gt_frames")

@dataclass
class OUTPUT_DIRECTORY:
    HOG: Path = Path("outputs/hog_frames")
    YOLO: Path = Path("outputs/yolo_frames")
    FASTER_RCNN: Path = Path("outputs/faster_rcnn_frames")

@dataclass
class DETECTION_TYPE:
    HOG: str = "HOG"
    YOLO: str = "YOLO"
    FASTER_RCNN: str = "FASTER_RCNN"

# ----- MODELS -----
YOLO_MODEL_12X = YOLO("yolo_models/yolo12x.pt")
YOLO_MODEL_8N = YOLO("yolo_models/yolov8n.pt")
YOLO_MODEL_8S = YOLO("yolo_models/yolov8s.pt")
FASTER_RCNN_MODEL = fasterrcnn_resnet50_fpn(pretrained=True)

# ---- Colors ----
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_PURPLE = (255, 0, 255)
COLOR_PINK = (203, 192, 255)

COLORS = [
    COLOR_RED,
    COLOR_ORANGE,
    COLOR_YELLOW,
    COLOR_GREEN,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_PURPLE,
    COLOR_PINK
]

# TODO: Requirements:
# For at least 50 frames, record the true number of people in each frame. You may optionally annotate bounding boxes for a subset of frames.
TRUE_PEOPLE_COUNT = {
    "frames/seq_000051.jpg": 17,
    "frames/seq_000058.jpg": 22,
    "frames/seq_000062.jpg": 18,
    "frames/seq_000066.jpg": 24,
    "frames/seq_000072.jpg": 26,
    "frames/seq_000076.jpg": 20,
    "frames/seq_000080.jpg": 17,
    "frames/seq_000085.jpg": 16,
    "frames/seq_000091.jpg": 18,
    "frames/seq_000094.jpg": 18,
    "frames/seq_000100.jpg": 31,
    "frames/seq_000107.jpg": 31,
    "frames/seq_000112.jpg": 23,
    "frames/seq_000118.jpg": 27,
    "frames/seq_000123.jpg": 25,
    "frames/seq_000127.jpg": 32,
    "frames/seq_000132.jpg": 29,
    "frames/seq_000138.jpg": 26,
    "frames/seq_000144.jpg": 26,
    "frames/seq_000150.jpg": 24,
    "frames/seq_000157.jpg": 24,
    "frames/seq_000163.jpg": 26,
    "frames/seq_000168.jpg": 23,
    "frames/seq_000174.jpg": 18,
    "frames/seq_000181.jpg": 20,
    "frames/seq_000187.jpg": 25,
    "frames/seq_000193.jpg": 22,
    "frames/seq_000198.jpg": 24,
    "frames/seq_000205.jpg": 23,
    "frames/seq_000212.jpg": 21,
    "frames/seq_000219.jpg": 24,
    "frames/seq_000225.jpg": 21,
    "frames/seq_000231.jpg": 20,
    "frames/seq_000236.jpg": 28,
    "frames/seq_000242.jpg": 25,
    "frames/seq_000247.jpg": 18,
    "frames/seq_000253.jpg": 19,
    "frames/seq_000259.jpg": 21,
    "frames/seq_000264.jpg": 25,
    "frames/seq_000270.jpg": 29,
    "frames/seq_000276.jpg": 28,
    "frames/seq_000283.jpg": 22,
    "frames/seq_000289.jpg": 26,
    "frames/seq_000294.jpg": 31,
    "frames/seq_000300.jpg": 28,
    "frames/seq_000307.jpg": 23,
    "frames/seq_000314.jpg": 16,
    "frames/seq_000320.jpg": 13,
    "frames/seq_000333.jpg": 19,
    "frames/seq_000347.jpg": 27
}

DETECTION_PRECISION_SUBSET = [58, 72, 80, 100, 112, 118, 138, 157, 168, 174, 193, 212, 231, 236, 247, 264, 283, 300, 320, 347]