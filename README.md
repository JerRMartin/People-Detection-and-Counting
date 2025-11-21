# People-Detection-and-Counting
> This is an assignment from Computer Vision (CAP-5415)

The goal of this project is to design and implement a people detection and counting system that works on low-quality images or videos (CCTV-style videos). To achive this we plan to detect humans in each frame (image), count the number of people, and evaluate performance.


## Setup 
Ensure Python's package installer (PIP) is installed on your machine. 

Open a terminal in the working directory (This is using a [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) terminal)

1. Create a virtual environment named `virtual_env`

```
python3 -m venv virtual_env
```

2. Activate the virtual environment
```
source virtual_env/bin/activate
```
> Note: You should now see your working directory prefixed by `(virtual_env)`


3. Install the requirements using pip
```
pip install -r requirements.txt
```

4. To run the program
```
python3 main.py
```
