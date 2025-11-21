from pathlib import Path

# Folders
FRAMES_DIR = Path("frames")
DATASET_DIR = "data"
ZIP_PATH = "data/mall_dataset.zip"


# TODO: Requirements:
# For at least 50 frames, record the true number of people in each frame. You may optionally annotate bounding boxes for a subset of frames.
TRUE_PEOPLE_COUNT = {
    "frames/seq_000088.jpg": 12,
}