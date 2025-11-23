from pathlib import Path

# Folders
RAW_FRAMES_DIR = Path("frames")
PROCESSED_FRAMES_DIR = Path("processed_frames")

FRAME_TYPE = tuple[RAW_FRAMES_DIR, PROCESSED_FRAMES_DIR]

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
    "frames/seq_00051.jpg": 17,
    "frames/seq_00058.jpg": 22,
    "frames/seq_00062.jpg": 18,
    "frames/seq_00066.jpg": 24,
    "frames/seq_00072.jpg": 26,
    "frames/seq_00076.jpg": 20,
    "frames/seq_00080.jpg": 17,
    "frames/seq_00085.jpg": 16,
    "frames/seq_00091.jpg": 18,
    "frames/seq_00094.jpg": 18,
    "frames/seq_00100.jpg": 31,
    "frames/seq_00107.jpg": 31,
    "frames/seq_00112.jpg": 23,
    "frames/seq_00118.jpg": 27,
    "frames/seq_00123.jpg": 25,
    "frames/seq_00127.jpg": 32,
    "frames/seq_00132.jpg": 33,
    "frames/seq_00138.jpg": 36,
    "frames/seq_00144.jpg": 37,
    "frames/seq_00150.jpg": 38,
    "frames/seq_00157.jpg": 31,
    "frames/seq_00163.jpg": 30,
    "frames/seq_00168.jpg": 33,
    "frames/seq_00174.jpg": 33,
    "frames/seq_00181.jpg": 31,
    "frames/seq_00187.jpg": 29,
    "frames/seq_00193.jpg": 31,
    "frames/seq_00198.jpg": 33,
    "frames/seq_00205.jpg": 37,
    "frames/seq_00212.jpg": 32,
    "frames/seq_00219.jpg": 30,
    "frames/seq_00225.jpg": 32,
    "frames/seq_00231.jpg": 34,
    "frames/seq_00236.jpg": 32,
    "frames/seq_00242.jpg": 25,
    "frames/seq_00247.jpg": 18,
    "frames/seq_00253.jpg": 19,
    "frames/seq_00259.jpg": 21,
    "frames/seq_00264.jpg": 25,
    "frames/seq_00270.jpg": 29,
    "frames/seq_00276.jpg": 28,
    "frames/seq_00283.jpg": 22,
    "frames/seq_00289.jpg": 26,
    "frames/seq_00294.jpg": 31,
    "frames/seq_00300.jpg": 28,
    "frames/seq_00307.jpg": 23,
    "frames/seq_00314.jpg": 16,
    "frames/seq_00320.jpg": 13,
    "frames/seq_00333.jpg": 19,
    "frames/seq_00347.jpg": 27
}