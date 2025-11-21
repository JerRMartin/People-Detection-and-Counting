# People-Detection-and-Counting
> This is an assignment from Computer Vision (CAP-5415)

The goal of this project is to design and implement a people detection and counting system that works on low-quality images or videos (CCTV-style videos). To achive this we plan to detect humans in each frame (image), count the number of people, and evaluate performance.


## Running the Project
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

### 3. Run the package entrypoint.

```bash
python -m src.main
# OR
python src/main.py
```

### Notes for Windows native (PowerShell / CMD)

- The above instructions assume you are running inside WSL. If you prefer to run natively on Windows (PowerShell), create and activate a venv with:

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

CMD:

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m pytest -q
```

## Project Data
We are using the [Mall Dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html). 

For convienence, we included **300 frames** from this dataset in the `/frames` directory within this project. 

## Example Command(s)
### TODO: Add these