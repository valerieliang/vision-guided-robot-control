# Vision-Guided Robot Control

This repository is starting with Milestone 1: a minimal perception pipeline that reads webcam frames, detects hand landmarks, draws them on screen, and can optionally record landmarks to a JSONL file.

Robot control, inverse kinematics, retargeting, and simulation are intentionally out of scope for this milestone.

## Setup

This project assumes Python 3.11.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Milestone 1

```bash
python main.py
```

Press `q` in the camera window to exit.

Settings are in `config.yaml`. Set `record_landmarks: true` to write timestamped landmark records to `output_path`.
