# People-Detection-and-Counting
> This is an assignment from Computer Vision (CAP-5415)

The goal of this project is to design and implement a people detection and counting system that works on low-quality images or videos (CCTV-style videos). To achive this we plan to detect humans in each frame (image), count the number of people, and evaluate performance.


## Preparing the Environment
### 1. Create a virtual environment named `.venv` (safe local name) and activate it.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Notes:
- If `python3` isn't available, try `python`. Check the Python version with `python --version`.

### 2. Install the project requirements.

```bash
pip install -r requirements.txt
```

### 2.5 Download the Models 
From [`ultralytics`](https://docs.ultralytics.com/models/yolo12/#detection-performance-coco-val2017) you need to download the following models and add them into the `yolo_models` folder in the project. 

1. `yolov8s.pt`
2. `yolov8n.pt`
3. `yolo12x.pt`


### Notes for Windows native (PowerShell / CMD)

- The above instructions assume you are running inside WSL. If you prefer to run natively on Windows (PowerShell), create and activate a venv with:

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CMD:

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Project Data
We are using the [Mall Dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html). 

For convienence, we included **300 frames** from this dataset in the `/frames` directory within this project. 

## Running the Project

### Preprocess all frames
```
python -m src.main preprocess
```
### Show a specific frame (both raw and processed versions)
```
python -m src.main show-frame --frame-number 250
```
> !NOTE: To exit a pop-up frame window, press `q`

### Show all frames
```
python -m src.main show-all-frames
```
> !NOTE: To exit ALL pop-up frame windows, press `q`

### Run ground truth comparison for a specific detection method
```
python -m src.main ground-truth --detection-type YOLO
python -m src.main ground-truth --detection-type FASTER_RCNN
python -m src.main ground-truth --detection-type HOG
```

### Run the complete pipeline (preprocessing + showing example frame + all ground truth comparisons)
```
python -m src.main full-pipeline
```
> !NOTE: To exit a pop-up frame window, press `q`