"""
Unified undistortion tool for camera calibration.

Supports both:
- Image sequences (PNG/JPG in folders) -> lossless video output
- Video files -> high-quality HEVC video output

Input structure:
    input/step2_undistortion/
    ├── cam01_take1.mp4           # Video files (camXX_*.ext)
    ├── cam01_take2.mov
    ├── cam02_calibration/        # OR image folders (camXX_*/camXX_*)
    │   ├── cam02_Pos00.png
    │   └── cam02_Pos01.png

Output:
    output/step2_undistortion/
    ├── cam01_take1.mp4           # Undistorted videos
    ├── cam02_calibration.mp4     # Undistorted video from images
    └── all_cams_undistorted.py   # Camera settings Python file
"""

import argparse
import warnings
import os
import platform
import subprocess
import cv2 as cv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
import re


# Disable OpenCL to avoid cache warnings on macOS
cv.ocl.setUseOpenCL(False)


# =============================================================================
# Intrinsics Loading
# =============================================================================

def load_yaml_file(file_path):
    """Load calibration data from a YAML file."""
    camID = None
    data = {}

    file_path_str = str(file_path)
    fs = cv.FileStorage(file_path_str, cv.FILE_STORAGE_READ)

    if fs.isOpened():
        data["K"] = fs.getNode("K").mat()
        data["ImgSizeWH"] = fs.getNode("ImgSizeWH").mat().flatten()
        data["DistCoeffs"] = fs.getNode("DistCoeffs").mat().flatten()
        camID = fs.getNode("CameraId").string()
        fs.release()
    else:
        print(f"Error: Could not open file {file_path}")

    return camID, data


def load_intrinsics(dir_intrinsics):
    """Load all intrinsic calibration files from a directory."""
    filename = "calibration.yaml"
    intrinsics = {}

    root_path = Path(dir_intrinsics)
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Intrinsics directory not found: {root_path}")

    for file_path in root_path.rglob(filename):
        camID, data = load_yaml_file(file_path)
        if camID:
            intrinsics[camID] = data

    if len(intrinsics) == 0:
        raise ValueError(f"No intrinsics found in {dir_intrinsics}.")

    return intrinsics


def has_distortion(dist_coeffs, threshold=1e-9):
    """Check if distortion coefficients are non-zero."""
    if dist_coeffs is None:
        return False
    return np.any(np.abs(dist_coeffs) > threshold)


def update_intrinsics(intr, resolution_target):
    """Scale intrinsics to match target resolution."""
    resolution_intr = intr["ImgSizeWH"]
    w_intr, h_intr = resolution_intr[0], resolution_intr[1]
    w_target, h_target = resolution_target[0], resolution_target[1]

    if w_intr != w_target or h_intr != h_target:
        scale_factor_w = w_target / w_intr
        scale_factor_h = h_target / h_intr

        if not np.isclose(scale_factor_w, scale_factor_h, rtol=1e-3):
            raise ValueError(
                f"Incompatible aspect ratios: intrinsic {w_intr}x{h_intr} vs target {w_target}x{h_target}"
            )

        if scale_factor_w > 1.0:
            warnings.warn(
                f"Upscaling detected ({w_intr}x{h_intr} -> {w_target}x{h_target}), this may affect accuracy.",
                UserWarning
            )

        intr = intr.copy()
        intr["K"] = np.array(intr["K"]) * scale_factor_w
        intr["K"][2, 2] = 1.0
        intr["ImgSizeWH"] = np.array(resolution_target)

    return intr


# =============================================================================
# Camera ID Extraction
# =============================================================================

def extract_cam_id(name):
    """Extract camera ID (camXX) from a filename or folder name."""
    match = re.match(r"(cam\d+)", name.lower())
    if match:
        return match.group(1)
    raise ValueError(
        f"Cannot extract camID from '{name}'. Expected format: camXX_* (e.g., cam01_take1.mp4)"
    )


# =============================================================================
# FFmpeg Encoder Detection
# =============================================================================

def get_available_encoder():
    """Detect the best available HEVC encoder for the current platform."""
    system = platform.system()

    # Define encoder priority per platform
    if system == "Darwin":  # macOS
        encoder_priority = ["hevc_videotoolbox", "libx265"]
    elif system == "Windows":
        encoder_priority = ["hevc_nvenc", "hevc_qsv", "libx265"]
    else:  # Linux
        encoder_priority = ["hevc_nvenc", "hevc_vaapi", "libx265"]

    # Check which encoders are available
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10
        )
        available_encoders = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Warning: ffmpeg not found or not responding, falling back to libx265")
        return "libx265"

    for encoder in encoder_priority:
        if encoder in available_encoders:
            return encoder

    return "libx265"  # Always available as fallback


def get_encoder_options(encoder):
    """Get encoder-specific options for high quality output."""
    if encoder == "hevc_videotoolbox":
        # macOS hardware encoder - uses bitrate mode (20Mbps = high quality)
        return ["-c:v", encoder, "-b:v", "20M", "-tag:v", "hvc1"]
    elif encoder == "hevc_nvenc":
        # NVIDIA GPU - CQ 18 is high quality
        return ["-c:v", encoder, "-preset", "p7", "-rc", "vbr", "-cq", "18", "-profile:v", "main"]
    elif encoder == "hevc_qsv":
        # Intel QuickSync
        return ["-c:v", encoder, "-preset", "veryslow", "-global_quality", "18"]
    elif encoder == "hevc_vaapi":
        # Linux VAAPI
        return ["-c:v", encoder, "-qp", "18"]
    else:
        # libx265 CPU fallback - CRF 18 is high quality
        return ["-c:v", "libx265", "-preset", "medium", "-crf", "18"]


def get_lossless_encoder_options():
    """Get encoder options for lossless output (for image sequences)."""
    # Use libx264 with crf 0 for true lossless in mp4 container
    return ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow"]


# =============================================================================
# Input Discovery
# =============================================================================

def discover_inputs(input_dir):
    """
    Discover all video files and image folders in the input directory.

    Returns:
        dict: {camID: [list of (type, path) tuples]}
              type is 'video' or 'image_folder'
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
    image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

    inputs_by_cam = {}

    for item in input_path.iterdir():
        if item.is_file() and item.suffix.lower() in video_extensions:
            # Video file
            try:
                cam_id = extract_cam_id(item.name)
                if cam_id not in inputs_by_cam:
                    inputs_by_cam[cam_id] = []
                inputs_by_cam[cam_id].append(("video", item))
            except ValueError as e:
                print(f"Skipping {item.name}: {e}")

        elif item.is_dir():
            # Check if it's an image folder
            images = [f for f in item.iterdir()
                     if f.is_file() and f.suffix.lower() in image_extensions]
            if images:
                try:
                    cam_id = extract_cam_id(item.name)
                    if cam_id not in inputs_by_cam:
                        inputs_by_cam[cam_id] = []
                    inputs_by_cam[cam_id].append(("image_folder", item))
                except ValueError as e:
                    print(f"Skipping folder {item.name}: {e}")

    return inputs_by_cam


# =============================================================================
# Undistortion Processing
# =============================================================================

def create_undistort_maps(intr):
    """Create undistortion remap arrays."""
    K = intr["K"]
    dist = intr["DistCoeffs"]
    size = tuple(map(int, intr["ImgSizeWH"]))

    mapx, mapy = cv.initUndistortRectifyMap(
        K, dist, None, K, size, cv.CV_16SC2
    )
    return mapx, mapy


def process_video(video_path, output_path, intr, encoder, encoder_options, fps=None):
    """Process a video file: undistort and re-encode."""
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv.CAP_PROP_FPS) if fps is None else fps

    # Update intrinsics for this resolution
    intr_scaled = update_intrinsics(intr, (w, h))
    mapx, mapy = create_undistort_maps(intr_scaled)

    # Build FFmpeg command
    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(video_fps),
        "-i", "-",
        "-an",
    ] + encoder_options + [
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        for _ in tqdm(range(total_frames), desc=f"  {video_path.name}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
            undistorted = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
            try:
                proc.stdin.write(undistorted.tobytes())
            except BrokenPipeError:
                # FFmpeg crashed - get the error message
                break
    finally:
        cap.release()
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"FFmpeg error:\n{stderr}")

    return output_path


def process_image_folder(folder_path, output_path, intr, fps=1):
    """Process an image folder: undistort images and create lossless video."""
    image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    images = natsorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not images:
        raise FileNotFoundError(f"No images found in {folder_path}")

    # Read first image to get resolution
    first_frame = cv.imread(str(images[0]))
    if first_frame is None:
        raise IOError(f"Cannot read image: {images[0]}")

    h, w = first_frame.shape[:2]

    # Update intrinsics for this resolution
    intr_scaled = update_intrinsics(intr, (w, h))
    mapx, mapy = create_undistort_maps(intr_scaled)

    # Use lossless encoding for images
    encoder_options = get_lossless_encoder_options()

    # Build FFmpeg command
    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-an",
    ] + encoder_options + [
        "-pix_fmt", "yuv444p",  # Better quality for lossless
        str(output_path)
    ]

    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        for img_path in tqdm(images, desc=f"  {folder_path.name}", leave=False):
            frame = cv.imread(str(img_path))
            if frame is None:
                warnings.warn(f"Cannot read {img_path}, skipping")
                continue
            undistorted = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
            try:
                proc.stdin.write(undistorted.tobytes())
            except BrokenPipeError:
                break
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"FFmpeg error:\n{stderr}")

    return output_path


# =============================================================================
# Output Generation
# =============================================================================

def get_intrinsic_parameters(intr):
    """Extract intrinsic parameters as a dictionary."""
    return {
        'fx': float(intr['K'][0, 0]),
        'fy': float(intr['K'][1, 1]),
        'cx': float(intr['K'][0, 2]),
        'cy': float(intr['K'][1, 2]),
        'k1': float(intr['DistCoeffs'][0]),
        'k2': float(intr['DistCoeffs'][1]),
        'p1': float(intr['DistCoeffs'][2]),
        'p2': float(intr['DistCoeffs'][3]),
        'k3': float(intr['DistCoeffs'][4]) if len(intr['DistCoeffs']) > 4 else 0.0,
    }


def generate_all_cams_undistorted_py(output_dir, processed_cameras):
    """
    Generate all_cams_undistorted.py file with CameraSettings for processed cameras.

    Args:
        output_dir: Directory to save the file
        processed_cameras: List of dicts with camera info
    """
    if not processed_cameras:
        return

    # Sort by camID
    processed_cameras = sorted(processed_cameras, key=lambda x: x['camID'])

    lines = []
    lines.append("from dataclasses import dataclass")
    lines.append("from typing import List")
    lines.append("")
    lines.append("")
    lines.append("@dataclass")
    lines.append("class CameraSettings:")
    lines.append("    model: str")
    lines.append("    resolution_x: int")
    lines.append("    resolution_y: int")
    lines.append("    focal_length_x: float")
    lines.append("    focal_length_y: float")
    lines.append("    principal_point_x: float")
    lines.append("    principal_point_y: float")
    lines.append("    distortion_coeffs: List[float]")
    lines.append("")
    lines.append("")

    cam_ids = []
    for cam in processed_cameras:
        cam_id = cam['camID']
        cam_ids.append(cam_id)

        dist_coeffs = cam['distortion_coeffs']
        dist_str = "[" + ", ".join(str(c) for c in dist_coeffs) + "]"

        lines.append(f"{cam_id} = CameraSettings(")
        lines.append(f"    model='generic',")
        lines.append(f"    resolution_x={cam['resolution_x']},")
        lines.append(f"    resolution_y={cam['resolution_y']},")
        lines.append(f"    focal_length_x={cam['focal_length_x']},")
        lines.append(f"    focal_length_y={cam['focal_length_y']},")
        lines.append(f"    principal_point_x={cam['principal_point_x']},")
        lines.append(f"    principal_point_y={cam['principal_point_y']},")
        lines.append(f"    distortion_coeffs={dist_str},")
        lines.append(")")
        lines.append("")

    # Generate dict_sdk
    lines.append("dict_sdk = {")
    for cam_id in cam_ids:
        lines.append(f"    '{cam_id}': {cam_id},")
    lines.append("}")
    lines.append("")

    # Write file
    output_path = Path(output_dir) / "all_cams_undistorted.py"
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved camera settings to {output_path}")


# =============================================================================
# Main Processing
# =============================================================================

def process_all(intrinsics, input_dir, output_dir, image_fps=1):
    """Process all inputs (videos and image folders)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover inputs
    inputs_by_cam = discover_inputs(input_dir)
    if not inputs_by_cam:
        raise ValueError(f"No valid inputs found in {input_dir}")

    # Detect encoder and get options with good defaults
    encoder = get_available_encoder()
    encoder_options = get_encoder_options(encoder)
    print(f"Using encoder: {encoder}")

    # Track processed cameras for output file
    processed_cameras = []
    skipped_cameras = []

    # Process each camera
    for cam_id in sorted(inputs_by_cam.keys()):
        items = inputs_by_cam[cam_id]

        # Check if we have intrinsics for this camera
        if cam_id not in intrinsics:
            print(f"Warning: No intrinsics found for {cam_id}, skipping")
            continue

        intr = intrinsics[cam_id]

        # Check if this camera has distortion
        if not has_distortion(intr["DistCoeffs"]):
            print(f"Skipping {cam_id}: no distortion (coefficients are zero)")
            skipped_cameras.append(cam_id)
            continue

        print(f"\nProcessing {cam_id} ({len(items)} item(s))...")

        for input_type, item_path in items:
            if input_type == "video":
                output_file = output_path / f"{item_path.stem}.mp4"
                try:
                    process_video(item_path, output_file, intr, encoder, encoder_options)
                    print(f"  Created: {output_file.name}")
                except Exception as e:
                    print(f"  Error processing {item_path.name}: {e}")
                    continue

            elif input_type == "image_folder":
                output_file = output_path / f"{item_path.name}.mp4"
                try:
                    process_image_folder(item_path, output_file, intr, fps=image_fps)
                    print(f"  Created: {output_file.name}")
                except Exception as e:
                    print(f"  Error processing {item_path.name}: {e}")
                    continue

        # Collect camera info for output file
        params = get_intrinsic_parameters(intr)
        processed_cameras.append({
            'camID': cam_id,
            'resolution_x': int(intr['ImgSizeWH'][0]),
            'resolution_y': int(intr['ImgSizeWH'][1]),
            'focal_length_x': params['fx'],
            'focal_length_y': params['fy'],
            'principal_point_x': params['cx'],
            'principal_point_y': params['cy'],
            'distortion_coeffs': [params['k1'], params['k2'], params['p1'], params['p2'], params['k3']],
        })

    # Generate output Python file
    generate_all_cams_undistorted_py(output_dir, processed_cameras)

    # Summary
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Processed cameras: {len(processed_cameras)}")
    if skipped_cameras:
        print(f"  Skipped (no distortion): {', '.join(skipped_cameras)}")
    print(f"  Output directory: {output_dir}")


def main():
    script_path = Path(__file__).parent
    os.chdir(script_path)

    default_dir_input = script_path / "input" / "step2_undistortion"
    default_dir_output = script_path / "output" / "step2_undistortion"
    default_dir_intrinsics = script_path / "output" / "step1_intrinsics"

    parser = argparse.ArgumentParser(
        description="Undistortion tool - processes videos and image folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python step2_undistortion.py

Input structure:
  input/step2_undistortion/
  ├── cam01_take1.mp4         # Video files
  ├── cam02_calibration/      # Image folders
  │   ├── cam02_Pos00.png
  │   └── cam02_Pos01.png
        """
    )
    parser.add_argument(
        "--path_dir_input",
        type=str,
        default=str(default_dir_input),
        help="Directory containing input videos/image folders",
    )
    parser.add_argument(
        "--path_dir_output",
        type=str,
        default=str(default_dir_output),
        help="Directory to store undistorted outputs",
    )
    parser.add_argument(
        "--path_dir_intrinsics",
        type=str,
        default=str(default_dir_intrinsics),
        help="Directory containing intrinsic calibration results from step1",
    )
    parser.add_argument(
        "--image_fps",
        type=int,
        default=1,
        help="Frame rate for videos created from image sequences (default: 1)",
    )
    args = parser.parse_args()

    # Load intrinsics
    print(f"Loading intrinsics from: {args.path_dir_intrinsics}")
    intrinsics = load_intrinsics(args.path_dir_intrinsics)
    print(f"Found intrinsics for: {', '.join(sorted(intrinsics.keys()))}")

    # Process all inputs
    process_all(
        intrinsics,
        args.path_dir_input,
        args.path_dir_output,
        image_fps=args.image_fps,
    )


if __name__ == "__main__":
    main()
