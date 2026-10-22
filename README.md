# Real-Time PPE Detection System

A real-time Personal Protective Equipment (PPE) detection demo built with Python, OpenCV, and Ultralytics YOLO. It processes webcam streams or video files, draws selected PPE detections, counts PPE-present and missing-PPE detections, and can optionally save annotated video.

## Features

- Real-time webcam or video-file inference
- Displays only six PPE classes: `Hardhat`, `NO-Hardhat`, `Mask`, `NO-Mask`, `Safety Vest`, and `NO-Safety Vest`
- Green boxes for PPE-present classes and red boxes for missing-PPE classes
- On-screen FPS and detection counters
- Configurable confidence threshold, inference size, and compute device
- Violation alerts after a configurable number of consecutive frames
- Optional sound alerts and annotated-video recording

> **Limitation:** Counters represent detected PPE bounding boxes, not unique workers. The script does not associate PPE boxes with individual people, so a detected PPE item alone does not prove that a particular worker is wearing it.

## Project structure

```text
ppe-detection/
├── ppe_detection.py
├── best.pt
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

- Python 3.10 or 3.11 recommended
- A webcam for live detection, or a local video file
- CPU inference; a compatible NVIDIA GPU may also be used if PyTorch and CUDA are configured

## Setup

Open a terminal in the project folder and run:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`.

## Run

### Webcam

```bash
python ppe_detection.py
```

### Video file

```bash
python ppe_detection.py --source path/to/video.mp4
```

### Save annotated output

```bash
python ppe_detection.py --source path/to/video.mp4 --save --output output.mp4
```

### Optional settings

```bash
python ppe_detection.py --conf 0.5 --imgsz 640 --device auto
python ppe_detection.py --alert-frames 10 --sound
```

Press `q` in the video window to stop detection.

| Argument | Purpose | Default |
|---|---|---|
| `--model` | Path to YOLO model weights | `best.pt` |
| `--source` | Camera index or video path | `0` |
| `--save` | Save annotated video | Off |
| `--output` | Output video filename | `output.mp4` |
| `--conf` | Detection confidence threshold | `0.4` |
| `--imgsz` | Inference image size | `640` |
| `--device` | `auto`, `cpu`, or GPU index such as `0` | `auto` |
| `--alert-frames` | Consecutive violation frames before alert | `10` |
| `--sound` | Enable optional beep alert | Off |
| `--alert-cooldown` | Seconds between repeat alerts during a continuous violation | `0` |

## Model weights

The included `best.pt` file contains the trained model weights used by this script. If replacing it, ensure the new model uses compatible class labels. The script filters display output to the six PPE class names above, ignoring capitalization and hyphen/underscore differences.

## Notes

- Camera access and performance depend on your hardware and installed OpenCV/PyTorch builds.
- `requirements.txt` preserves the package versions from the supplied environment, converted to UTF-8 text for compatibility.
- This is a demonstration tool, not a substitute for workplace safety procedures or human supervision.

## Suggested GitHub topics

`computer-vision`, `object-detection`, `yolo`, `opencv`, `personal-protective-equipment`, `python`
