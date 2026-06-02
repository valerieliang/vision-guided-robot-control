# Vision-Guided Robot Control

Real-time hand and neurotech-driven control for an Allegro Hand simulation.

The active pipeline is:

```text
webcam hand landmarks -> smoothing -> retargeted hand angles -> Allegro controller -> PyBullet Allegro URDF visualization
```

Decoded neurotech data can also drive the same simulated hand:

```text
decoded EEG/EMG label -> hand-pose mapper -> Allegro controller -> PyBullet Allegro URDF visualization
```

## Setup

Python 3.10 is recommended on macOS. The visualization is the PyBullet GUI
loading `simulation/assets/allegro/allegro_hand_right.urdf`, matching the
previous commit's black Allegro hand view.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If `pybullet` fails through pip on macOS, use Conda for that dependency:

```bash
conda create -n robot-control python=3.10 -y
conda activate robot-control
conda install -c conda-forge pybullet -y
python -m pip install opencv-python pyyaml numpy mediapipe pytest ruff
```

## Run

```bash
python main.py
```

Press `q` in the camera window to exit. Closing the PyBullet GUI also stops the
loop.

## Input Modes

Set `input_source` in `config.yaml`.

If left/right labels look swapped because your camera preview is mirrored, set:

```yaml
handedness_display: mirrored
```

The default is:

```yaml
handedness_display: mediapipe
```

```yaml
input_source: hand
```

Uses webcam MediaPipe landmarks, temporal smoothing, retargeting, and the
Allegro simulation.

```yaml
input_source: neurotech_replay
neuro_replay_path: data/neuro_replay_example.jsonl
```

Uses decoded labels from a `.jsonl` or `.csv` file. Rows need `label` or
`intent`, plus optional `confidence`.

```yaml
input_source: synthetic_neurotech
```

Runs a no-hardware decoded-label demo.

```yaml
input_source: hybrid
neuro_replay_path: data/neuro_replay_example.jsonl
```

Uses hand tracking when a hand is visible, then falls back to decoded
neurotech labels when the hand drops out.

Supported decoded labels include:

- `rest`
- `open`
- `close`
- `pinch`
- `point`
- `left_hand`
- `right_hand`
- `feet`
- `tongue`

## Smoothing

The hand was shaky because raw MediaPipe landmarks were previously passed
directly into retargeting. Current smoothing knobs in `config.yaml`:

```yaml
landmark_smoothing_alpha: 0.45
hand_hold_frames: 3
angle_smoothing_alpha: 0.35
angle_deadband_rad: 0.015
angle_max_step_rad: 0.22
```

Lower alpha values are smoother but laggier. Higher values are more responsive
but shakier.

## Public EEG Datasets

This repo does not vendor raw EEG datasets. Put local copies under:

```text
datasets/eeg/physionet_eegmmi/
datasets/eeg/bci_comp_iv_2a/
```

Supported dataset registry entries live in `neurotech/datasets.py`.

Useful public sources:

- PhysioNet EEG Motor Movement/Imagery: https://physionet.org/content/eegmmidb/1.0.0/
- BCI Competition IV Dataset 2a: https://www.bbci.de/competition/iv/

The intended workflow is:

1. Train or evaluate EEG decoders offline on public datasets.
2. Export decoded labels to `.jsonl` or `.csv`.
3. Replay those labels through `input_source: neurotech_replay`.

This keeps real EEG training separate from realtime robot control and avoids
claiming that synthetic labels are actual neural intent.

## Tests

```bash
python -m pytest
```
