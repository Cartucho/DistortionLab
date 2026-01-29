# DistortionLab

Camera lens calibration and undistortion pipeline with a focus on visual interpretability.

## Overview

DistortionLab provides tools for:
1. **Intrinsic calibration** - Estimate lens distortion from a chessboard calibration video
2. **Undistortion** - Remove lens distortion from videos and images

## Requirements

```bash
pip install -r requirements.txt
```

Additionally, [FFmpeg](https://ffmpeg.org/) must be installed and available in your PATH.

## Workflow

### Step 1: Intrinsic Calibration

Estimate camera intrinsics (focal length, principal point, distortion coefficients) from a calibration video.

**Input:** Video of a chessboard pattern moved through the frame
**Output:** Calibration YAML + distortion plot

```bash
python step1_intrinsics.py
```

Place calibration videos in `input/step1_intrinsics/` named as `camXX_*.mp4` (e.g., `cam01_calibration.mp4`).

Options:
- `--board_shape 5 8` - Chessboard size in squares (default: 5x8)
- `--desired_frames 200` - Number of frames to process (default: 200)
- `--no_distortion` - Assume zero distortion (only estimate focal length and principal point)

### Step 2: Undistortion

Remove lens distortion from videos or image sequences using the calibration from Step 1.

**Input:** Videos or image folders
**Output:** Undistorted videos (.mp4)

```bash
python step2_undistortion.py
```

Place inputs in `input/step2_undistortion/`:
- **Videos:** `camXX_*.mp4`, `camXX_*.mov`, etc.
- **Image folders:** `camXX_folder/` containing PNG/JPG images

The script automatically:
- Detects the best encoder for your platform (VideoToolbox on macOS, NVENC on NVIDIA, CPU fallback)
- Skips cameras with zero distortion coefficients
- Scales intrinsics if input resolution differs from calibration resolution

## Directory Structure

```
DistortionLab/
├── input/
│   ├── step1_intrinsics/        # Calibration videos
│   │   └── cam01_calibration.mp4
│   │   └── cam02_calibration.mp4
│   └── step2_undistortion/      # Videos/images to undistort
│       ├── cam01_take1.mp4
│       ├── cam01_take2.mp4
│       ├── cam01_take3.mp4
│       └── cam02_images/
│           ├── frame_001.png
│           └── frame_002.png
├── output/
│   ├── step1_intrinsics/        # Calibration results
│   │   ├── cam01_calibration/
│   │   │   ├── calibration.yaml
│   │   │   ├── distortion_plot.png
│   │   │   └── coverage.png
│   │   ├── cam02_calibration/
│   │   │   ├── ...
│   │   └── all_cams.py
│   └── step2_undistortion/      # Undistorted outputs
│       ├── cam01_take1.mp4
│       ├── cam01_take2.mp4
│       ├── cam01_take3.mp4
│       ├── cam02_images.mp4
│       └── all_cams_undistorted.py
├── step1_intrinsics.py
├── step2_undistortion.py
└── requirements.txt
```

## Output Format

Both steps generate a Python file with camera settings that can be imported directly:

```python
from output.step1_intrinsics.all_cams import cam01, dict_sdk

print(cam01.focal_length_x)
print(cam01.distortion_coeffs)
```
