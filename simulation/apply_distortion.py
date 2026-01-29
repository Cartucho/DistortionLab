"""
Apply lens distortion to a video.

This script takes an undistorted video and applies camera lens distortion
using the standard 5-parameter distortion model (k1, k2, p1, p2, k3).

The distortion model follows OpenCV's convention:
- k1, k2, k3: Radial distortion coefficients
- p1, p2: Tangential distortion coefficients

Usage:
    1. (Optional) Export camera intrinsics from Blender using blender_export_intrinsics.py
    2. Set parameters below
    3. Run: python apply_distortion.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.linalg import lstsq


# ============================================================================
# PARAMETERS - Adjust these to customize the distortion
# ============================================================================

# Input/Output paths
INPUT_VIDEO = "cam01_undistorted.mp4"
OUTPUT_VIDEO = "cam01_distorted.mp4"

# Camera intrinsics file (exported from Blender via blender_export_intrinsics.py)
# Set to None to estimate from video resolution
INTRINSICS_FILE = "sim_intrinsics.yaml"

# Distortion coefficients (k1, k2, p1, p2, k3)
# Positive k1/k2 = barrel distortion (edges bow outward)
# Negative k1/k2 = pincushion distortion (edges bow inward)
K1 = 0.35
K2 = 0.15
P1 = 0.02
P2 = 0.02
K3 = 0.01


# ============================================================================
# INTRINSICS LOADING
# ============================================================================

def load_intrinsics_yaml(yaml_path):
    """
    Load camera intrinsics from YAML file.
    Simple parser - no external dependencies.
    """
    intrinsics = {}
    with open(yaml_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Try to parse as number
                try:
                    if '.' in value:
                        intrinsics[key] = float(value)
                    else:
                        intrinsics[key] = int(value)
                except ValueError:
                    intrinsics[key] = value
    return intrinsics


def get_camera_matrix(width, height, intrinsics_file=None):
    """
    Get camera matrix K from intrinsics file or estimate from resolution.

    Returns:
        K: 3x3 camera matrix
        source: String describing where intrinsics came from
    """
    if intrinsics_file and Path(intrinsics_file).exists():
        intr = load_intrinsics_yaml(intrinsics_file)
        fx = intr.get('fx', width)
        fy = intr.get('fy', width)
        cx = intr.get('cx', width / 2)
        cy = intr.get('cy', height / 2)
        source = f"Loaded from {intrinsics_file}"
    else:
        # Estimate: focal length ~ width, principal point = center
        fx = fy = float(width)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        source = "Estimated from resolution"

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return K, source


# ============================================================================
# DISTORTION VISUALIZATION
# ============================================================================

def visualize_distortion(K, D, height, width, output_path, nstep=20, contour_levels=10):
    """
    Generate and save a distortion plot showing the distortion field.

    Args:
        K: Camera matrix (3x3)
        D: Distortion coefficients [k1, k2, p1, p2, k3]
        height: Image height
        width: Image width
        output_path: Path to save the plot
        nstep: Grid step size for visualization
        contour_levels: Number of contour levels
    """
    # Ensure D has 14 elements (pad with zeros)
    d = np.zeros(14)
    D = np.array(D).ravel()
    d[:len(D)] = D
    D = d

    k1, k2, p1, p2, k3 = D[0], D[1], D[2], D[3], D[4]

    # Create grid of pixel coordinates
    u, v = np.meshgrid(
        np.arange(0, width, nstep),
        np.arange(0, height, nstep)
    )

    # Convert to homogeneous coordinates
    b = np.array([
        u.ravel(),
        v.ravel(),
        np.ones(u.size)
    ])

    # Project to normalized coordinates
    xyz = lstsq(K, b, rcond=None)[0]

    xp = xyz[0, :] / xyz[2, :]
    yp = xyz[1, :] / xyz[2, :]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3

    # Apply distortion model
    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2 + 2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2 + 2*yp**2) + 2*p2*(xp*yp) + D[10]*r2 + D[11]*r4

    # Convert back to pixel coordinates
    u2 = K[0, 0]*xpp + K[0, 2]
    v2 = K[1, 1]*ypp + K[1, 2]

    # Calculate displacement
    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du, dv).reshape(u.shape)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Quiver plot showing displacement vectors
    ax.quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue", alpha=0.7)

    # Mark image center and principal point
    ax.plot(width/2, height/2, "x", markersize=10, label="Image center")
    ax.plot(K[0, 2], K[1, 2], "^", markersize=10, label="Principal point")

    # Contour plot showing magnitude of distortion
    CS = ax.contour(u, v, dr, colors="black", levels=contour_levels)
    ax.clabel(CS, inline=1, fontsize=8, fmt='%.0f px')

    ax.set_aspect('equal', 'box')
    ax.set_title(f"Distortion Model\nk1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, p1={p1:.4f}, p2={p2:.4f}")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_ylim(max(v.ravel()), 0)  # Flip y-axis
    ax.legend(loc='upper right')

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Distortion plot saved to: {output_path}")


# ============================================================================
# DISTORTION APPLICATION
# ============================================================================

def create_distortion_map(width, height, K, dist_coeffs):
    """
    Create distortion map for remapping.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        K: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]

    Returns:
        map1, map2: Distortion maps for cv2.remap()
    """
    # Create pixel coordinates
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)

    # Convert to normalized coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_norm = (x_coords - cx) / fx
    y_norm = (y_coords - cy) / fy

    # Calculate r^2
    r2 = x_norm**2 + y_norm**2
    r4 = r2**2
    r6 = r2**3

    # Distortion coefficients
    k1, k2, p1, p2, k3 = dist_coeffs

    # Radial distortion
    radial = 1 + k1*r2 + k2*r4 + k3*r6

    # Tangential distortion
    x_distorted = x_norm*radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
    y_distorted = y_norm*radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm

    # Convert back to pixel coordinates
    x_distorted = x_distorted * fx + cx
    y_distorted = y_distorted * fy + cy

    return x_distorted.astype(np.float32), y_distorted.astype(np.float32)


def apply_distortion_to_video(input_path, output_path, dist_coeffs, intrinsics_file=None):
    """
    Apply lens distortion to a video.

    Args:
        input_path: Path to input undistorted video
        output_path: Path to save distorted video
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
        intrinsics_file: Path to intrinsics YAML (optional)
    """
    # Open input video
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    print(f"\nInput Video: {input_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Get camera matrix
    K, source = get_camera_matrix(width, height, intrinsics_file)

    print(f"\nCamera Intrinsics ({source}):")
    print(f"  Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f} pixels")
    print(f"  Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    print(f"\nDistortion Coefficients:")
    print(f"  k1={dist_coeffs[0]:.6f}, k2={dist_coeffs[1]:.6f}, k3={dist_coeffs[4]:.6f}")
    print(f"  p1={dist_coeffs[2]:.6f}, p2={dist_coeffs[3]:.6f}")

    # Create distortion maps
    print("\nCreating distortion maps...")
    map1, map2 = create_distortion_map(width, height, K, dist_coeffs)

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate distortion plot
    plot_path = output_path.parent / f"{output_path.stem}_distortion_plot.png"
    visualize_distortion(K, dist_coeffs, height, width, plot_path)

    # Remove existing output file if present
    if output_path.exists():
        output_path.unlink()

    # Create video writer
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    print(f"\nApplying distortion to video...")

    # Process frames
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply distortion using remap
        distorted_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        out.write(distorted_frame)

        frame_count += 1
        pbar.update(1)

    pbar.close()

    # Cleanup
    cap.release()
    out.release()

    print(f"\nDistortion applied successfully!")
    print(f"  Output: {output_path}")
    print(f"  Frames processed: {frame_count}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create distortion coefficient array
    distortion_coeffs = np.array([K1, K2, P1, P2, K3], dtype=np.float64)

    # Check if intrinsics file exists
    intrinsics_file = INTRINSICS_FILE if Path(INTRINSICS_FILE).exists() else None

    if intrinsics_file:
        print(f"Using camera intrinsics from: {intrinsics_file}")
    else:
        print("No intrinsics file found - will estimate from video resolution")

    # Apply distortion
    try:
        apply_distortion_to_video(
            input_path=INPUT_VIDEO,
            output_path=OUTPUT_VIDEO,
            dist_coeffs=distortion_coeffs,
            intrinsics_file=intrinsics_file,
        )

        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Check the distortion plot to verify the applied distortion")
        print("2. Use the distorted video as input for step1_intrinsics.py")
        print("3. Compare the recovered distortion coefficients with ground truth")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
