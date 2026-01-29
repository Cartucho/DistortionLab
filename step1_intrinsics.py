import argparse
import warnings
import os
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm
import multiprocessing
from path import Path
import re
import matplotlib.pyplot as plt
from numpy.linalg import lstsq


def validate_homography(corners, pattern_size, img_diagonal, threshold_ratio=0.00136):
    """
    Check if corners are geometrically consistent using homography.
    A valid chessboard detection should have all corners lying on a plane.

    Args:
        corners: Detected corner points
        pattern_size: (cols, rows) inner corners
        img_diagonal: Image diagonal in pixels (for scaling threshold)
        threshold_ratio: Threshold as ratio of image diagonal (default 0.00136)
                        Calibrated so 1080p (~2203px diagonal) gets ~3px (OpenCV default).
                        For 8K (diagonal ~8789px) this gives ~12px threshold
                        For 4K (diagonal ~4406px) this gives ~6px threshold

    Returns (is_valid, inlier_ratio, reprojection_error)
    """
    threshold = img_diagonal * threshold_ratio

    # Create ideal object points (2D grid)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 2), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    img_pts = corners.reshape(-1, 2).astype(np.float32)

    # Find homography with RANSAC
    H, mask = cv.findHomography(objp, img_pts, cv.RANSAC, threshold)

    if H is None:
        return False, 0.0, float('inf')

    inlier_ratio = np.sum(mask) / len(mask)

    # Calculate mean reprojection error for inliers
    projected = cv.perspectiveTransform(objp.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(img_pts - projected, axis=1)
    mean_error = np.mean(errors[mask.ravel() == 1]) if np.sum(mask) > 0 else float('inf')

    return inlier_ratio > 0.95, inlier_ratio, mean_error


def validate_color_pattern(corners, gray, pattern_size):
    """
    Verify that squares have alternating intensities (black/white pattern).
    Returns (is_valid, contrast_ratio)
    """
    corners_grid = corners.reshape(pattern_size[1], pattern_size[0], 2)

    intensities = []
    expected_colors = []

    for i in range(pattern_size[1] - 1):
        for j in range(pattern_size[0] - 1):
            # Get 4 corners of this square
            c1 = corners_grid[i, j]
            c2 = corners_grid[i, j + 1]
            c3 = corners_grid[i + 1, j]
            c4 = corners_grid[i + 1, j + 1]

            # Center of the square
            center = (c1 + c2 + c3 + c4) / 4
            cx, cy = int(center[0]), int(center[1])

            # Sample a small region around center
            if 2 <= cx < gray.shape[1] - 2 and 2 <= cy < gray.shape[0] - 2:
                patch = gray[cy-2:cy+3, cx-2:cx+3]
                intensities.append(np.mean(patch))
                expected_colors.append((i + j) % 2)

    if len(intensities) < 4:
        return False, 0.0

    intensities = np.array(intensities)
    expected_colors = np.array(expected_colors)

    group0 = intensities[expected_colors == 0]
    group1 = intensities[expected_colors == 1]

    if len(group0) == 0 or len(group1) == 0:
        return False, 0.0

    mean0, mean1 = np.mean(group0), np.mean(group1)
    std0, std1 = np.std(group0), np.std(group1)

    # Fisher's discriminant ratio
    separation = abs(mean0 - mean1)
    pooled_std = np.sqrt((std0**2 + std1**2) / 2) + 1e-6
    contrast_ratio = separation / pooled_std

    # Coefficient of variation for each group
    cv0 = std0 / (mean0 + 1e-6)
    cv1 = std1 / (mean1 + 1e-6)

    # Good detection: high contrast ratio, low internal variation
    is_valid = contrast_ratio > 2.0 and cv0 < 0.5 and cv1 < 0.5

    return is_valid, contrast_ratio


def validate_collinearity(corners, pattern_size, img_diagonal, threshold_ratio=0.001):
    """
    Check if corners in each row lie close to a straight line.
    This is especially important for no-distortion mode where we assume
    straight lines in the world project to straight lines in the image.

    Args:
        corners: Detected corner points
        pattern_size: (cols, rows) inner corners
        img_diagonal: Image diagonal in pixels (for scaling threshold)
        threshold_ratio: Threshold as ratio of image diagonal (default 0.001 = 0.1%)
                        For 8K (diagonal ~8789px) this gives ~8.8px threshold
                        For 4K (diagonal ~4406px) this gives ~4.4px threshold

    Returns (is_valid, max_deviation) where max_deviation is the maximum
    distance of any corner from its fitted line.
    """
    threshold = img_diagonal * threshold_ratio
    corners_grid = corners.reshape(pattern_size[1], pattern_size[0], 2)
    max_deviation = 0.0

    # Check each row
    for row_idx in range(pattern_size[1]):
        row_points = corners_grid[row_idx]  # shape: (pattern_size[0], 2)

        # Fit a line using least squares: ax + by + c = 0
        # We use SVD to find the best fit line
        centroid = row_points.mean(axis=0)
        centered = row_points - centroid

        # SVD gives us the principal components
        _, _, vh = np.linalg.svd(centered)
        # The line direction is the first principal component
        # The normal to the line is the second principal component
        normal = vh[1]  # Normal vector to the line

        # Calculate distance of each point to the line
        # Distance = |dot(point - centroid, normal)|
        distances = np.abs(np.dot(centered, normal))
        row_max_dev = distances.max()
        max_deviation = max(max_deviation, row_max_dev)

    # Also check each column
    for col_idx in range(pattern_size[0]):
        col_points = corners_grid[:, col_idx]  # shape: (pattern_size[1], 2)

        centroid = col_points.mean(axis=0)
        centered = col_points - centroid

        _, _, vh = np.linalg.svd(centered)
        normal = vh[1]

        distances = np.abs(np.dot(centered, normal))
        col_max_dev = distances.max()
        max_deviation = max(max_deviation, col_max_dev)

    is_valid = max_deviation < threshold
    return is_valid, max_deviation


def validate_detection(corners, gray, pattern_size):
    """
    Validate a chessboard detection using homography and color checks.
    Returns True if valid, False if likely a bad detection.
    """
    # Compute image diagonal for resolution-scaled thresholds
    img_diagonal = np.sqrt(gray.shape[1]**2 + gray.shape[0]**2)

    # Check 1: Homography consistency (geometric check)
    h_valid, _, _ = validate_homography(corners, pattern_size, img_diagonal)
    if not h_valid:
        return False

    # Check 2: Color pattern (intensity check)
    c_valid, _ = validate_color_pattern(corners, gray, pattern_size)
    if not c_valid:
        return False

    return True


def apply_gamma(gray, gamma):
    """Apply gamma correction. gamma < 1 brightens, gamma > 1 darkens."""
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv.LUT(gray, lut)


def detect_and_save_corners(frame, frame_path, pattern_size, save_debug_images=True):
    """Detect corners in a frame and save results. Uses cascading fallback for difficult images.

    Note: This function only performs OpenCV detection. Validation (homography, color pattern)
    is done later during calibration so parameters can be tuned without re-processing video.
    """
    pattern_size = tuple(pattern_size)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flags = cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY

    # Try detection on original image first
    valid, corners = cv.findChessboardCornersSB(gray, pattern_size, None, flags=flags)

    # Fallback: gamma correction for dark images (gamma < 1 brightens)
    if not valid:
        gray_bright = apply_gamma(gray, 0.5)
        valid, corners = cv.findChessboardCornersSB(gray_bright, pattern_size, None, flags=flags)

    if valid:
        np.save(f"{frame_path}_corners.npy", corners)
        if save_debug_images:
            cv.drawChessboardCorners(frame, pattern_size, corners, valid)
            cv.imwrite(f"{frame_path}.jpg", frame)
        return True
    return False


def process_batch_parallel(batch, pattern_size, n_workers, save_debug_images=True):
    """Process a batch of frames in parallel using thread pool."""
    from concurrent.futures import ThreadPoolExecutor

    def process_one(item):
        frame, frame_path = item
        return detect_and_save_corners(frame, frame_path, pattern_size, save_debug_images)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_one, batch))

    return sum(1 for r in results if r)


def load_corners(dir_calib_lens_output_data, objp):
    """Load previously saved corner data and their frame indices."""
    pts_obj, pts_img, frame_indices = [], [], []
    corner_files = sorted(Path(dir_calib_lens_output_data).glob("*_corners.npy"))
    for f in corner_files:
        corners = np.load(f)
        pts_img.append(corners)
        pts_obj.append(objp)
        # Extract frame index from filename (e.g., "0042_corners.npy" -> 42)
        frame_idx = int(os.path.basename(f).replace("_corners.npy", ""))
        frame_indices.append(frame_idx)
    return pts_obj, pts_img, frame_indices


def load_outliers(dir_calib_lens_output):
    """Load the list of outlier frame indices."""
    outliers_file = os.path.join(dir_calib_lens_output, "outliers.json")
    if os.path.exists(outliers_file):
        with open(outliers_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_outliers(dir_calib_lens_output, outliers):
    """Save the list of outlier frame indices."""
    outliers_file = os.path.join(dir_calib_lens_output, "outliers.json")
    with open(outliers_file, 'w') as f:
        json.dump(sorted(list(outliers)), f, indent=2)


def load_failed_frames(dir_calib_lens_output_data):
    """Load the list of frame indices where detection failed."""
    failed_file = os.path.join(dir_calib_lens_output_data, "failed_frames.json")
    if os.path.exists(failed_file):
        with open(failed_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_failed_frames(dir_calib_lens_output_data, failed_frames):
    """Save the list of frame indices where detection failed."""
    failed_file = os.path.join(dir_calib_lens_output_data, "failed_frames.json")
    with open(failed_file, 'w') as f:
        json.dump(sorted(list(failed_frames)), f, indent=2)


def compute_additional_frame_indices(existing_indices, total_frames, num_additional):
    """Compute frame indices that are maximally spaced from existing ones."""
    if num_additional <= 0:
        return []

    existing_set = set(existing_indices)
    all_candidates = set(range(total_frames)) - existing_set

    if len(all_candidates) <= num_additional:
        return sorted(list(all_candidates))

    # Greedy selection: pick frames that maximize minimum distance to existing/selected
    selected = []
    existing_arr = np.array(sorted(existing_indices)) if existing_indices else np.array([])

    for _ in range(num_additional):
        best_candidate = None
        best_min_dist = -1

        candidates = list(all_candidates - set(selected))
        if not candidates:
            break

        # Combine existing and already selected indices
        reference_arr = np.concatenate([existing_arr, np.array(selected)]) if selected else existing_arr

        if len(reference_arr) == 0:
            # No reference points, use uniform spacing
            uniform_indices = np.linspace(0, total_frames - 1, num_additional, dtype=int)
            return sorted([int(i) for i in uniform_indices if i in all_candidates][:num_additional])

        # Find candidate with maximum minimum distance to reference points
        for c in candidates:
            min_dist = np.min(np.abs(reference_arr - c))
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c

        if best_candidate is not None:
            selected.append(best_candidate)

    return sorted(selected)


def get_video_paths(path_dir_input):
    """Get valid video file paths from a directory."""
    video_paths = []
    for f in os.listdir(path_dir_input):
        file_path = os.path.join(path_dir_input, f)
        if os.path.isfile(file_path):
            cap = cv.VideoCapture(file_path)
            if cap.isOpened():
                video_paths.append(file_path)
                cap.release()

    if not video_paths:
        raise Exception(f"No video files found in {path_dir_input}.")

    return sorted(video_paths)


def get_calib_obj(args):
    """Generate object points for the calibration pattern."""
    pattern_size = tuple(args.pattern_size)
    objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    return objp


def calibrate(pts_obj, pts_img, vid_width, vid_height, no_distortion=False):
    """Perform camera calibration.

    Args:
        no_distortion: If True, fix all distortion coefficients to zero.
                       Only estimates focal length and principal point.
    """
    if no_distortion:
        # Fix all distortion coefficients to zero
        flags = (cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 |
                 cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6 |
                 cv.CALIB_ZERO_TANGENT_DIST)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            pts_obj, pts_img, (vid_width, vid_height), None, None, flags=flags
        )
    else:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            pts_obj, pts_img, (vid_width, vid_height), None, None
        )
    return mtx, dist, rvecs, tvecs


def recalculate_rvecs_tvecs(pts_obj, pts_img, mtx, dist):
    """Recalculate rvecs and tvecs for all images."""
    rvecs = []
    tvecs = []
    for objp, imgp in zip(pts_obj, pts_img):
        _, rvec, tvec = cv.solvePnP(objp, imgp, mtx, dist)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs


def calculate_reprojection_errors(pts_obj, pts_img, mtx, dist, rvecs, tvecs):
    """Calculate reprojection errors for all images."""
    errors = []
    for i, (objp, imgp) in enumerate(zip(pts_obj, pts_img)):
        imgp = imgp.reshape(-1, 2)
        reprojected_points, _ = cv.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
        reprojected_points = reprojected_points.reshape(-1, 2)
        error = np.sqrt(np.sum((imgp - reprojected_points) ** 2, axis=1)).max()
        errors.append(error)
    return errors


def calculate_coverage_score(pts_img, vid_width, vid_height, grid_size=8):
    """
    Calculate how uniformly corners cover the image using a grid-based approach.
    Returns a score between 0 and 1, where 1 means perfect uniform coverage.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    cell_width = vid_width / grid_size
    cell_height = vid_height / grid_size

    for corners in pts_img:
        for corner in corners:
            x, y = corner[0][0], corner[0][1]
            gx = min(int(x / cell_width), grid_size - 1)
            gy = min(int(y / cell_height), grid_size - 1)
            grid[gy, gx] += 1

    # Calculate uniformity: ratio of filled cells and entropy-like measure
    filled_cells = np.sum(grid > 0)
    total_cells = grid_size * grid_size
    fill_ratio = filled_cells / total_cells

    # Coefficient of variation (lower is more uniform)
    if grid.sum() > 0:
        mean_count = grid[grid > 0].mean() if filled_cells > 0 else 0
        std_count = grid[grid > 0].std() if filled_cells > 0 else 0
        cv = std_count / (mean_count + 1e-6)
        uniformity = 1.0 / (1.0 + cv)  # Higher uniformity = lower CV
    else:
        uniformity = 0

    # Combined score: prioritize filling cells, then uniformity
    score = 0.6 * fill_ratio + 0.4 * uniformity
    return score, grid


def calculate_pose_diversity(rvecs, tvecs):
    """
    Calculate pose diversity from rotation and translation vectors.
    Returns a score where higher means more diverse poses.
    """
    if len(rvecs) < 2:
        return 0.0

    # Convert to numpy arrays
    rvecs_arr = np.array([r.flatten() for r in rvecs])
    tvecs_arr = np.array([t.flatten() for t in tvecs])

    # Rotation diversity: variance in rotation angles
    # Convert rodrigues vectors to angles
    angles = np.linalg.norm(rvecs_arr, axis=1)
    angle_diversity = np.std(angles)

    # Also consider direction diversity (unit vectors)
    directions = rvecs_arr / (np.linalg.norm(rvecs_arr, axis=1, keepdims=True) + 1e-6)
    direction_spread = np.std(directions)

    # Translation diversity: variance in distances and positions
    distances = np.linalg.norm(tvecs_arr, axis=1)
    distance_diversity = np.std(distances) / (np.mean(distances) + 1e-6)  # Normalized

    # Position diversity
    position_diversity = np.mean(np.std(tvecs_arr, axis=0))

    # Combined score (normalized to roughly 0-1 range)
    score = (angle_diversity + direction_spread + distance_diversity + position_diversity / 100) / 4
    return min(score, 1.0)


def estimate_poses_for_selection(pts_obj, pts_img, vid_width, vid_height):
    """
    Estimate poses for all images using a simple calibration.
    Used for image selection before final calibration.
    """
    # Quick calibration to get approximate camera matrix
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        pts_obj, pts_img, (vid_width, vid_height), None, None,
        flags=cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_ZERO_TANGENT_DIST
    )
    return rvecs, tvecs


def select_optimal_subset(pts_obj, pts_img, frame_indices, vid_width, vid_height,
                          num_selected, no_distortion=False):
    """
    Select an optimal subset of images for calibration.

    For distortion estimation (default): prioritizes uniform coverage across the image
    For no-distortion mode: prioritizes pose diversity (angles and distances)

    Uses greedy selection to incrementally build a good subset.
    """
    n_images = len(pts_obj)
    if n_images <= num_selected:
        print(f"  Only {n_images} images available, using all")
        return pts_obj, pts_img, frame_indices

    print(f"  Selecting {num_selected} images from {n_images} candidates...")

    # Estimate poses for all images
    rvecs, tvecs = estimate_poses_for_selection(pts_obj, pts_img, vid_width, vid_height)

    # Greedy selection
    selected_indices = []
    remaining_indices = list(range(n_images))

    # Start with the image that has corners closest to image center (good anchor)
    center = np.array([vid_width / 2, vid_height / 2])
    best_start = None
    best_dist = float('inf')
    for i in remaining_indices:
        corners = pts_img[i].reshape(-1, 2)
        mean_corner = corners.mean(axis=0)
        dist_to_center = np.linalg.norm(mean_corner - center)
        if dist_to_center < best_dist:
            best_dist = dist_to_center
            best_start = i
    selected_indices.append(best_start)
    remaining_indices.remove(best_start)

    # Greedy selection based on mode
    while len(selected_indices) < num_selected and remaining_indices:
        best_candidate = None
        best_score = -float('inf')

        # Current subset data
        current_pts_img = [pts_img[i] for i in selected_indices]
        current_rvecs = [rvecs[i] for i in selected_indices]
        current_tvecs = [tvecs[i] for i in selected_indices]

        for candidate in remaining_indices:
            # Evaluate adding this candidate
            test_pts_img = current_pts_img + [pts_img[candidate]]
            test_rvecs = current_rvecs + [rvecs[candidate]]
            test_tvecs = current_tvecs + [tvecs[candidate]]

            coverage_score, _ = calculate_coverage_score(test_pts_img, vid_width, vid_height)
            pose_score = calculate_pose_diversity(test_rvecs, test_tvecs)

            if no_distortion:
                # Prioritize pose diversity, coverage is secondary
                score = 0.2 * coverage_score + 0.8 * pose_score
            else:
                # Prioritize coverage for distortion estimation
                score = 0.7 * coverage_score + 0.3 * pose_score

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is not None:
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)

    # Build selected data
    selected_pts_obj = [pts_obj[i] for i in selected_indices]
    selected_pts_img = [pts_img[i] for i in selected_indices]
    selected_frame_indices = [frame_indices[i] for i in selected_indices]

    # Report final scores
    final_coverage, grid = calculate_coverage_score(selected_pts_img, vid_width, vid_height)
    final_rvecs = [rvecs[i] for i in selected_indices]
    final_tvecs = [tvecs[i] for i in selected_indices]
    final_pose = calculate_pose_diversity(final_rvecs, final_tvecs)

    print(f"  Selection complete: coverage={final_coverage:.3f}, pose_diversity={final_pose:.3f}")
    print(f"  Grid coverage: {np.sum(grid > 0)}/64 cells filled")

    return selected_pts_obj, selected_pts_img, selected_frame_indices


def save_selected_images(dir_output, frame_indices, coverage_score, pose_score):
    """Save the list of selected images to a JSON file."""
    selection_file = os.path.join(dir_output, "selected_images.json")
    data = {
        "selected_frames": sorted(frame_indices),
        "num_selected": len(frame_indices),
        "coverage_score": float(coverage_score),
        "pose_diversity_score": float(pose_score)
    }
    with open(selection_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved selection to {selection_file}")


def get_calib_data(args, video_path, dir_calib_lens_output_data, dir_calib_lens_output,
                   save_debug_images=True, no_distortion=False, num_selected=50):
    """Extract calibration data from video or from .npy files."""
    objp = get_calib_obj(args)

    cap = cv.VideoCapture(video_path)
    vid_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if vid_width != 7680 or vid_height != 4320:
        msg = f"\n8K videos expected as input. Received {vid_width}x{vid_height} instead."
        warnings.warn(msg, UserWarning)

    # Load existing corners, outliers, and failed frames
    pts_obj, pts_img, frame_indices = load_corners(dir_calib_lens_output_data, objp)
    outliers = load_outliers(dir_calib_lens_output)
    failed_frames = load_failed_frames(dir_calib_lens_output_data)

    # Filter out outliers from existing data
    if outliers:
        filtered_data = [
            (po, pi, fi) for po, pi, fi in zip(pts_obj, pts_img, frame_indices)
            if fi not in outliers
        ]
        if filtered_data:
            pts_obj, pts_img, frame_indices = zip(*filtered_data)
            pts_obj, pts_img, frame_indices = list(pts_obj), list(pts_img), list(frame_indices)
        else:
            pts_obj, pts_img, frame_indices = [], [], []

    existing_count = len(pts_obj)
    desired_frames = args.desired_frames

    # Total attempted = successful + outliers + failed
    total_attempted = len(frame_indices) + len(outliers) + len(failed_frames)
    print(f"Found {existing_count} existing valid detections, {len(outliers)} outliers, {len(failed_frames)} failed")

    # Determine if we need more frames (only if we haven't attempted enough)
    if total_attempted < desired_frames:
        num_additional = desired_frames - total_attempted
        print(f"Need {num_additional} more frames to reach {desired_frames}")

        # Compute additional frame indices that are optimally spaced
        # Exclude all previously attempted frames
        all_processed_indices = set(frame_indices) | outliers | failed_frames
        frames_to_process = compute_additional_frame_indices(
            list(all_processed_indices), total_frames, num_additional
        )

        if frames_to_process:
            # Auto-detect number of CPU cores
            n_workers = multiprocessing.cpu_count()
            batch_size = n_workers * 2  # Read ahead to keep workers busy
            print(f"Using {n_workers} CPU cores, batch size {batch_size}")

            num_detected = 0
            batch = []

            print(f"Processing {len(frames_to_process)} frames...")
            for frame_idx in tqdm(frames_to_process, desc="Processing frames"):
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    path_data = os.path.join(dir_calib_lens_output_data, f"{frame_idx:04d}")
                    batch.append((frame, path_data))

                # Process batch when full
                if len(batch) >= batch_size:
                    num_detected += process_batch_parallel(batch, args.pattern_size, n_workers, save_debug_images)
                    batch = []

            # Process remaining frames
            if batch:
                num_detected += process_batch_parallel(batch, args.pattern_size, n_workers, save_debug_images)

            print(f"Detected corners in {num_detected}/{len(frames_to_process)} frames")

            # Load the newly detected corners (to get their frame indices)
            new_pts_obj_loaded, new_pts_img_loaded, new_frame_indices = load_corners(dir_calib_lens_output_data, objp)

            # Track which frames failed detection
            newly_detected_set = set(new_frame_indices)
            for fi in frames_to_process:
                if fi not in newly_detected_set:
                    failed_frames.add(fi)
            save_failed_frames(dir_calib_lens_output_data, failed_frames)

            # Merge: keep existing + add only the new ones
            existing_set = set(frame_indices)
            for po, pi, fi in zip(new_pts_obj_loaded, new_pts_img_loaded, new_frame_indices):
                if fi not in existing_set and fi not in outliers:
                    pts_obj.append(po)
                    pts_img.append(pi)
                    frame_indices.append(fi)

            print(f"Total valid detections after processing: {len(pts_obj)}")
    else:
        print(f"Already attempted {total_attempted} frames, no additional frames needed")

    cap.release()

    if len(pts_obj) < 25:
        raise ValueError(f"Only {len(pts_obj)} valid detections found. Need at least 25 for calibration.")

    # Validation phase: check geometric consistency of detected corners
    # Only run homography validation in no_distortion mode - with lens distortion,
    # corners won't fit a pure homography (projective transformation assumes no distortion)
    # See: "Radial Distortion Homography" (CVPR 2015) - ignoring distortion leads to wrong estimates
    if no_distortion:
        print(f"Validating {len(pts_obj)} detections (homography check)...")
        pattern_size = tuple(args.pattern_size)
        img_diagonal = np.sqrt(vid_width**2 + vid_height**2)
        validated_pts_obj, validated_pts_img, validated_frame_indices = [], [], []
        validation_rejected = 0
        for po, pi, fi in zip(pts_obj, pts_img, frame_indices):
            h_valid, inlier_ratio, reproj_err = validate_homography(pi, pattern_size, img_diagonal)
            if not h_valid:
                outliers.add(fi)
                validation_rejected += 1
                print(f"  Rejected frame {fi}: homography check failed (inlier_ratio={inlier_ratio:.2f}, error={reproj_err:.2f}px)")
            else:
                validated_pts_obj.append(po)
                validated_pts_img.append(pi)
                validated_frame_indices.append(fi)
        if validation_rejected > 0:
            save_outliers(dir_calib_lens_output, outliers)
            pts_obj, pts_img, frame_indices = validated_pts_obj, validated_pts_img, validated_frame_indices
            print(f"  {len(pts_obj)} images passed validation, {validation_rejected} rejected")
        else:
            print(f"  All {len(pts_obj)} images passed validation")

        if len(pts_obj) < 25:
            raise ValueError(f"Only {len(pts_obj)} valid detections after validation. Need at least 25 for calibration.")
    else:
        print(f"Skipping homography validation (lens distortion mode - will use reprojection error filtering instead)")

    # Collinearity check for no-distortion mode
    # In no-distortion mode, straight lines in the world should project to straight lines in the image
    if no_distortion:
        print("Running collinearity check (no-distortion mode)...")
        pattern_size = tuple(args.pattern_size)
        img_diagonal = np.sqrt(vid_width**2 + vid_height**2)
        collinear_pts_obj, collinear_pts_img, collinear_frame_indices = [], [], []
        rejected_count = 0
        total_count = len(pts_obj)
        for po, pi, fi in zip(pts_obj, pts_img, frame_indices):
            is_collinear, max_dev = validate_collinearity(pi, pattern_size, img_diagonal)
            if not is_collinear:
                outliers.add(fi)
                rejected_count += 1
                print(f"  Rejected frame {fi}: corners not collinear (max_deviation={max_dev:.2f}px)")
            else:
                collinear_pts_obj.append(po)
                collinear_pts_img.append(pi)
                collinear_frame_indices.append(fi)
        save_outliers(dir_calib_lens_output, outliers)
        pts_obj, pts_img, frame_indices = collinear_pts_obj, collinear_pts_img, collinear_frame_indices
        print(f"  {len(pts_obj)} images passed collinearity check")

        # Warn if many images failed collinearity - suggests lens distortion
        rejection_ratio = rejected_count / total_count if total_count > 0 else 0
        if rejection_ratio > 0.5:
            warnings.warn(
                f"\n*** WARNING: {rejection_ratio*100:.0f}% of images failed collinearity check. ***\n"
                f"This suggests significant lens distortion is present.\n"
                f"Consider running WITHOUT --no_distortion to estimate distortion parameters.\n",
                UserWarning
            )

        if len(pts_obj) < 25:
            raise ValueError(f"Only {len(pts_obj)} valid detections after collinearity check. Need at least 25.")

    # Initial calibration with all data (for outlier detection)
    print(f"Initial calibration with {len(pts_obj)} images...")
    mtx, dist, rvecs, tvecs = calibrate(pts_obj, pts_img, vid_width, vid_height, no_distortion)

    # Calculate reprojection errors
    errors = calculate_reprojection_errors(pts_obj, pts_img, mtx, dist, rvecs, tvecs)

    # Statistical outlier rejection: median + 3 * MAD, but with a minimum floor
    # that scales with resolution (avoids over-rejecting when calibration is already excellent)
    # Floor ratio calibrated so 1080p (~2203px diagonal) gets ~2px minimum
    img_diagonal = np.sqrt(vid_width**2 + vid_height**2)
    min_threshold_ratio = 0.0009  # ~2px for 1080p, ~4px for 4K, ~8px for 8K
    min_threshold = img_diagonal * min_threshold_ratio
    median_err = np.median(errors)
    mad = np.median(np.abs(errors - median_err))
    threshold = max(median_err + 3 * mad, min_threshold)
    print(f"Outlier threshold: {threshold:.2f}px (median={median_err:.2f}, MAD={mad:.2f}, min_floor={min_threshold:.2f}px)")

    # Mark high-error detections as outliers
    refined_pts_obj, refined_pts_img, refined_frame_indices = [], [], []
    for po, pi, fi, err in zip(pts_obj, pts_img, frame_indices, errors):
        if err > threshold:
            outliers.add(fi)
            print(f"  Auto-rejected frame {fi} (error={err:.2f}px > {threshold:.2f}px)")
        else:
            refined_pts_obj.append(po)
            refined_pts_img.append(pi)
            refined_frame_indices.append(fi)
    save_outliers(dir_calib_lens_output, outliers)

    if len(refined_pts_obj) < 25:
        raise ValueError(f"Only {len(refined_pts_obj)} valid detections after outlier removal. Need at least 25.")

    # Image selection: select optimal subset for final calibration
    print(f"\nImage selection phase ({len(refined_pts_obj)} candidates)...")
    mode_str = "no-distortion (prioritizing pose diversity)" if no_distortion else "distortion (prioritizing coverage)"
    print(f"  Mode: {mode_str}")

    selected_pts_obj, selected_pts_img, selected_frame_indices = select_optimal_subset(
        refined_pts_obj, refined_pts_img, refined_frame_indices,
        vid_width, vid_height, num_selected, no_distortion
    )

    # Final calibration with selected data
    print(f"\nFinal calibration with {len(selected_pts_obj)} selected images...")
    mtx, dist, rvecs, tvecs = calibrate(selected_pts_obj, selected_pts_img, vid_width, vid_height, no_distortion)

    # Calculate final metrics
    final_coverage, _ = calculate_coverage_score(selected_pts_img, vid_width, vid_height)
    final_pose = calculate_pose_diversity(rvecs, tvecs)
    final_errors = calculate_reprojection_errors(selected_pts_obj, selected_pts_img, mtx, dist, rvecs, tvecs)

    # Save selected images list
    save_selected_images(dir_calib_lens_output, selected_frame_indices, final_coverage, final_pose)

    # Save calibration metrics
    metrics = {
        "num_images": len(selected_pts_obj),
        "coverage_score": float(final_coverage),
        "pose_diversity_score": float(final_pose),
        "reprojection_error": {
            "mean": float(np.mean(final_errors)),
            "median": float(np.median(final_errors)),
            "std": float(np.std(final_errors)),
            "max": float(np.max(final_errors)),
            "min": float(np.min(final_errors))
        },
        "no_distortion_mode": no_distortion
    }
    metrics_file = os.path.join(dir_calib_lens_output, "calibration_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Reprojection error: mean={metrics['reprojection_error']['mean']:.3f}px, "
          f"max={metrics['reprojection_error']['max']:.3f}px")

    return selected_pts_obj, selected_pts_img, vid_width, vid_height, mtx, dist


def draw_coverage_image(dir_calib_lens_output, pts_img, vid_width, vid_height, grid_size=8):
    """Draw and save an image showing the coverage of the detected corners.

    Shows an 8x8 grid with cells colored by coverage:
    - Red = no coverage
    - Green = has coverage (brighter = more points)
    """
    # Use the same coverage calculation as calculate_coverage_score
    _, grid = calculate_coverage_score(pts_img, vid_width, vid_height, grid_size)

    cell_width = vid_width / grid_size
    cell_height = vid_height / grid_size

    # Create the image
    im = np.zeros((vid_height, vid_width, 3), dtype=np.uint8)

    # Draw grid cells with color based on coverage
    max_count = grid.max() if grid.max() > 0 else 1

    for gy in range(grid_size):
        for gx in range(grid_size):
            x1 = int(gx * cell_width)
            y1 = int(gy * cell_height)
            x2 = int((gx + 1) * cell_width)
            y2 = int((gy + 1) * cell_height)

            count = grid[gy, gx]
            if count == 0:
                # No coverage - red
                color = (40, 40, 180)  # BGR - dark red
            else:
                # Has coverage - green, intensity based on density
                intensity = int(60 + 140 * (count / max_count))  # Range 60-200
                color = (40, intensity, 40)  # BGR - green

            cv.rectangle(im, (x1, y1), (x2, y2), color, -1)  # Filled rectangle

    # Draw grid lines
    for i in range(grid_size + 1):
        x = int(i * cell_width)
        cv.line(im, (x, 0), (x, vid_height), (100, 100, 100), 1)
        y = int(i * cell_height)
        cv.line(im, (0, y), (vid_width, y), (100, 100, 100), 1)

    # Add coverage stats text
    filled_cells = np.sum(grid > 0)
    total_cells = grid_size * grid_size
    text = f"Coverage: {filled_cells}/{total_cells} cells ({100*filled_cells/total_cells:.0f}%)"
    cv.putText(im, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Draw white rectangle around the image margin
    cv.rectangle(im, (0, 0), (vid_width - 1, vid_height - 1), (255, 255, 255), 2)

    # Draw corner points on top (bright green dots)
    for corners in pts_img:
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            cv.circle(im, (x, y), 4, (0, 255, 0), -1)

    cv.imwrite(str(Path(dir_calib_lens_output) / "coverage.png"), im)


def save_calibration_results(filename, cam_calib_final):
    """Save calibration results to a file."""
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    for key, value in cam_calib_final.items():
        fs.write(key, value)
    fs.release()


def generate_all_cams_py(output_dir, calibration_results):
    """Generate all_cams.py file with CameraSettings for all calibrated cameras.

    Args:
        output_dir: Directory to save all_cams.py
        calibration_results: List of dicts with keys: camID, resolution_x, resolution_y,
                           focal_length_x, focal_length_y, principal_point_x,
                           principal_point_y, distortion_coeffs
    """
    if not calibration_results:
        return

    # Sort by camID for consistent ordering
    calibration_results = sorted(calibration_results, key=lambda x: x['camID'])

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
    for result in calibration_results:
        cam_id = result['camID']
        cam_ids.append(cam_id)

        # Format distortion coefficients
        dist_coeffs = result['distortion_coeffs']
        dist_str = "[" + ", ".join(f"{c}" for c in dist_coeffs) + "]"

        lines.append(f"{cam_id} = CameraSettings(")
        lines.append(f"    model='generic',")
        lines.append(f"    resolution_x={result['resolution_x']},")
        lines.append(f"    resolution_y={result['resolution_y']},")
        lines.append(f"    focal_length_x={result['focal_length_x']},")
        lines.append(f"    focal_length_y={result['focal_length_y']},")
        lines.append(f"    principal_point_x={result['principal_point_x']},")
        lines.append(f"    principal_point_y={result['principal_point_y']},")
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
    output_path = os.path.join(output_dir, "all_cams.py")
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved all camera settings to {output_path}")


def save_lens_calib(
    args, dir_calib_lens_output, camID, pts_obj, pts_img, vid_width, vid_height, camera_matrix, dist_coeffs
):
    """Perform camera calibration and save the results."""
    cam_calib_final = {
        "K": camera_matrix,
        "ImgSizeWH": np.array([vid_width, vid_height]),
        "DistCoeffs": dist_coeffs.ravel(),
        "CameraId": camID,
    }
    np.set_printoptions(precision=4, suppress=True)
    print("Results:")
    print(f"{cam_calib_final['K']}")
    filename = os.path.join(dir_calib_lens_output, f"calibration.yaml")
    save_calibration_results(filename, cam_calib_final)


def extract_cam_id(f_name, allow_non_cam_videos=False):
    # Use a regular expression to match the pattern 'cam' followed by digits, optionally followed by '_'
    match = re.match(r"(cam\d+)_?", f_name.lower())
    if match:
        camID = match.group(1)  # This captures the 'camXX' part
    else:
        if allow_non_cam_videos:
            camID = os.path.splitext(f_name)[0]
        else:
            raise ValueError(
                "File name should start with the camID 'camXX', where XX is a number. 'camXX' or 'camXX_' expected."
            )
    return camID


def visualizeDistortion(K, D, h, w, camID, output_path, contour_levels=10, nstep=20):
    """Generate and save a distortion plot for the camera."""
    # Extract camera parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pad distortion coefficients to 14 elements
    D = D.ravel()
    d = np.zeros(14)
    d[:D.size] = D
    D = d
    k1, k2, p1, p2, k3 = D[0], D[1], D[2], D[3], D[4]

    # Create grid of pixel coordinates
    u, v = np.meshgrid(
        np.arange(0, w, nstep),
        np.arange(0, h, nstep)
    )

    # Convert to homogeneous coordinates and project to normalized coords
    b = np.array([u.ravel(), v.ravel(), np.ones(u.size)])
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
    u2 = fx*xpp + cx
    v2 = fy*ypp + cy

    # Calculate displacement
    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du, dv).reshape(u.shape)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Quiver plot showing displacement vectors
    ax.quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue", alpha=0.7)

    # Mark image center and principal point
    ax.plot(w/2, h/2, "x", markersize=10, label="Image center")
    ax.plot(cx, cy, "^", markersize=10, label=f"Principal point ({cx:.1f}, {cy:.1f})")

    # Contour plot showing magnitude of distortion
    CS = ax.contour(u, v, dr, colors="black", levels=contour_levels)
    ax.clabel(CS, inline=1, fontsize=8, fmt='%.0f px')

    ax.set_aspect('equal', 'box')
    ax.set_title(f"{camID} Distortion Model\nk1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, p1={p1:.4f}, p2={p2:.4f}")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_ylim(max(v.ravel()), 0)  # Flip y-axis
    ax.legend(loc='upper right')

    # Save
    plt.tight_layout()
    out_img = os.path.join(output_path, "distortion_plot.png")
    plt.savefig(out_img, dpi=150)
    plt.close(fig)


def main():
    script_path = Path(__file__).parent
    os.chdir(script_path)

    default_dir_input = os.path.join(script_path, "input", "step1_intrinsics")
    default_dir_output = os.path.join(script_path, "output", "step1_intrinsics")

    parser = argparse.ArgumentParser(description="Lens Calibration Tool")
    parser.add_argument(
        "--path_dir_input",
        type=str,
        default=default_dir_input,
        help="Directory containing input videos for lens calibration.",
    )
    parser.add_argument(
        "--board_shape",
        nargs=2,
        type=int,
        default=[5, 8],
        help="Number of squares on the chessboard [cols rows] (default: 5 8). "
             "Internally converted to inner corners (e.g. 5x8 squares -> 4x7 inner corners).",
    )
    parser.add_argument(
        "--desired_frames", type=int, default=200, help="Number of frames to use"
    )
    parser.add_argument(
        "--no_debug_images",
        action="store_true",
        help="Skip saving debug images with drawn corners (only save .npy files)",
    )
    parser.add_argument(
        "--no_distortion",
        action="store_true",
        help="Assume zero lens distortion (fix all distortion coefficients to 0). "
             "Prioritizes pose diversity over edge coverage for image selection.",
    )
    parser.add_argument(
        "--num_selected",
        type=int,
        default=50,
        help="Number of images to select for final calibration (default: 50)",
    )
    parser.add_argument(
        "--allow_non_cam_videos",
        action="store_true",
        help="Allow processing videos that do not start with 'cam', using the video name as camID.",
    )
    args = parser.parse_args()
    args.path_dir_output = default_dir_output  # Directory to save the lens calibration results.

    # Convert board shape (squares) to pattern size (inner corners)
    # e.g., 8x5 squares -> 7x4 inner corners
    if args.board_shape[0] < 2 or args.board_shape[1] < 2:
        print("Error: --board_shape must be at least 2x2 squares")
        exit()
    args.pattern_size = [args.board_shape[1] - 1, args.board_shape[0] - 1]
    print(f"Board shape: {args.board_shape[0]}x{args.board_shape[1]} squares -> {args.pattern_size[0]}x{args.pattern_size[1]} inner corners")

    if args.desired_frames < 25:
        print("Error: --desired_frames must be at least 25 for intrinsic calibration")
        exit()
    if args.num_selected < 25:
        print("Error: --num_selected must be at least 25 for intrinsic calibration")
        exit()
    if args.num_selected > args.desired_frames:
        print(f"Error: --num_selected ({args.num_selected}) cannot be larger than --desired_frames ({args.desired_frames})")
        exit()

    video_paths = get_video_paths(args.path_dir_input)

    # Collect calibration results for all cameras
    all_calibration_results = []

    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        camID = extract_cam_id(video_name, args.allow_non_cam_videos)
        if not args.allow_non_cam_videos:
            if not camID.startswith("cam") or not camID[3:].isdigit():
                raise ValueError(
                    "File name should start with the camID 'camXX_', where XX is a number. 'camXX' expected."
                )

        video_name_base = video_name.split(".")[0]

        dir_calib_lens_output = os.path.join(args.path_dir_output, video_name_base)
        dir_calib_lens_output_data = os.path.join(
            dir_calib_lens_output, "lens_calib_data"
        )
        calibration_file = os.path.join(dir_calib_lens_output, "calibration.yaml")

        # Check if we can skip this camera
        if os.path.exists(calibration_file):
            # Check if user wants more frames than we've already attempted
            objp = get_calib_obj(args)
            existing_pts_obj, _, existing_frame_indices = load_corners(dir_calib_lens_output_data, objp)
            existing_outliers = load_outliers(dir_calib_lens_output)
            existing_failed = load_failed_frames(dir_calib_lens_output_data)
            # Total attempted = successful detections + outliers + failed
            total_attempted = len(existing_frame_indices) + len(existing_failed)
            valid_count = len([fi for fi in existing_frame_indices if fi not in existing_outliers])

            if total_attempted >= args.desired_frames:
                print(f"========== Skipping {video_name}, camID:{camID} (attempted {total_attempted} frames, {valid_count} valid) ==========")
                # Load existing calibration data for all_cams.py
                fs = cv.FileStorage(calibration_file, cv.FILE_STORAGE_READ)
                K = fs.getNode("K").mat()
                img_size = fs.getNode("ImgSizeWH").mat().ravel()
                dist = fs.getNode("DistCoeffs").mat().ravel()
                fs.release()
                all_calibration_results.append({
                    'camID': camID,
                    'resolution_x': int(img_size[0]),
                    'resolution_y': int(img_size[1]),
                    'focal_length_x': float(K[0, 0]),
                    'focal_length_y': float(K[1, 1]),
                    'principal_point_x': float(K[0, 2]),
                    'principal_point_y': float(K[1, 2]),
                    'distortion_coeffs': [float(c) for c in dist[:5]],
                })
                continue
            else:
                print(f"========== Re-processing {video_name}, camID:{camID} (attempted {total_attempted}, want {args.desired_frames}) ==========")
        else:
            print(f"========== Processing {video_name}, camID:{camID} ==========")
            # Clear stale calibration outputs when re-doing calibration
            # Corner data (.npy) and failed_frames.json are kept (detection is independent)
            # But outliers and calibration outputs should be fresh
            stale_files = [
                os.path.join(dir_calib_lens_output, "outliers.json"),
                os.path.join(dir_calib_lens_output, "distortion_plot.png"),
                os.path.join(dir_calib_lens_output, "coverage.png"),
                os.path.join(dir_calib_lens_output, "calibration_metrics.json"),
                os.path.join(dir_calib_lens_output, "selected_images.json"),
            ]
            for f in stale_files:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"  Cleared stale file: {os.path.basename(f)}")

        os.makedirs(dir_calib_lens_output_data, exist_ok=True)

        pts_obj, pts_img, vid_width, vid_height, camera_matrix, dist_coeffs = get_calib_data(
            args, video_path, dir_calib_lens_output_data, dir_calib_lens_output,
            save_debug_images=not args.no_debug_images,
            no_distortion=args.no_distortion,
            num_selected=args.num_selected
        )

        save_lens_calib(
            args, dir_calib_lens_output, camID, pts_obj, pts_img, vid_width, vid_height, camera_matrix, dist_coeffs
        )

        draw_coverage_image(dir_calib_lens_output, pts_img, vid_width, vid_height)

        # Generate and save distortion plot (skip if no_distortion mode)
        if not args.no_distortion:
            visualizeDistortion(camera_matrix, dist_coeffs, vid_height, vid_width, camID, dir_calib_lens_output)
        else:
            print("  Skipping distortion plot (no_distortion mode)")

        # Collect calibration results for all_cams.py
        all_calibration_results.append({
            'camID': camID,
            'resolution_x': vid_width,
            'resolution_y': vid_height,
            'focal_length_x': float(camera_matrix[0, 0]),
            'focal_length_y': float(camera_matrix[1, 1]),
            'principal_point_x': float(camera_matrix[0, 2]),
            'principal_point_y': float(camera_matrix[1, 2]),
            'distortion_coeffs': [float(c) for c in dist_coeffs.ravel()[:5]],
        })

    # Generate all_cams.py with all calibration results
    generate_all_cams_py(args.path_dir_output, all_calibration_results)


if __name__ == "__main__":
    main()
