"""
Blender script to generate random keyframe poses for a calibration board.

This script positions a board object at random poses visible from a camera,
creating an animated sequence suitable for camera calibration simulation.

Usage in Blender:
    Run this script in Blender's scripting workspace or via command line:
    blender sim.blend --python generate_board_animation.py
"""

import bpy
import math
import random
from mathutils import Vector, Euler


# ============================================================================
# PARAMETERS - Adjust these to customize the animation
# ============================================================================

# Object names
CAMERA_NAME = "Camera"
BOARD_NAME = "board"

# Animation parameters
N_KEYFRAMES = 10          # Number of keyframe poses
FRAMES_BETWEEN = 50       # Number of frames (M) to interpolate between keyframes

# Position constraints (positive values = distance in front of camera)
Z_MIN = 1.0               # Minimum distance from camera (closer)
Z_MAX = 4.0               # Maximum distance from camera (farther)

# Rotation constraints (in degrees)
# These control how much the board can tilt/rotate while still showing its front face
# Smaller values = board faces camera more directly, larger = more variation
ROTATION_X_RANGE = 45     # Pitch: tilt up/down (±degrees)
ROTATION_Y_RANGE = 45     # Yaw: turn left/right (±degrees)
ROTATION_Z_RANGE = 180    # Roll: rotate in-plane (±degrees) - doesn't affect visibility

# View coverage - how much of the camera's FOV the board should cover
# (min, max) as fraction of FOV. E.g., (0.4, 0.7) means board fills 40-70% of view
# Higher values = board stays more centered and visible
VIEW_COVERAGE = (0.5, 0.7)

# Random seed for reproducibility (set to None for different results each time)
RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_camera_fov(camera_obj):
    """Calculate the camera's field of view in radians."""
    camera_data = camera_obj.data

    if camera_data.type != 'PERSP':
        print("Warning: Camera is not perspective. Results may be unexpected.")

    # Get the effective field of view
    if camera_data.sensor_fit == 'VERTICAL':
        fov = camera_data.angle
    else:  # HORIZONTAL or AUTO
        fov = camera_data.angle

    return fov


def calculate_position_range(camera_obj, z_distance, coverage_factor=0.5):
    """
    Calculate reasonable X, Y position ranges for given Z distance.

    Args:
        camera_obj: The camera object
        z_distance: Distance along camera's Z axis
        coverage_factor: How much of the FOV the board should occupy (0-1)

    Returns:
        (x_range, y_range) as the maximum offset from center
    """
    fov = get_camera_fov(camera_obj)

    # Calculate the view width at this distance
    # For a perspective camera: width = 2 * tan(fov/2) * distance
    view_width_at_z = 2 * math.tan(fov / 2) * abs(z_distance)

    # The board can offset from center, but not too much
    # Use a smaller multiplier to keep the board more centered and visible
    # coverage_factor here means "how much of the FOV the board fills"
    # We reduce position range to ~20-30% of view width to ensure visibility
    position_range = view_width_at_z * (1.0 - coverage_factor) * 0.4

    return position_range


def set_board_pose(board_obj, camera_obj, z_distance, coverage, rotation_ranges, frame_num):
    """
    Set the board to a random pose visible from the camera.

    Args:
        board_obj: The board object to position
        camera_obj: The camera object (for reference frame)
        z_distance: Positive distance in front of camera (in Blender units)
        coverage: Tuple (min, max) for view coverage fraction
        rotation_ranges: Tuple of (x_range, y_range, z_range) in degrees
        frame_num: Frame number to set the keyframe
    """
    # Get camera's world matrix
    cam_matrix = camera_obj.matrix_world

    # Random coverage factor between min and max
    coverage_factor = random.uniform(coverage[0], coverage[1])

    # Calculate position range for this Z distance
    pos_range = calculate_position_range(camera_obj, z_distance, coverage_factor)

    # Generate random local position (in camera space)
    x_local = random.uniform(-pos_range, pos_range)
    y_local = random.uniform(-pos_range, pos_range)

    # Convert from camera local coordinates to world coordinates
    # In Blender camera space: +X is right, +Y is up, -Z is forward
    # So we use -z_distance to place the board in front of the camera
    local_pos = Vector((x_local, y_local, -z_distance))
    world_pos = cam_matrix @ local_pos

    # Set position
    board_obj.location = world_pos
    board_obj.keyframe_insert(data_path="location", frame=frame_num)

    # Unpack rotation ranges
    rot_x_range, rot_y_range, rot_z_range = rotation_ranges

    # Generate random rotation (in radians) with separate ranges per axis
    rot_x = math.radians(random.uniform(-rot_x_range, rot_x_range))
    rot_y = math.radians(random.uniform(-rot_y_range, rot_y_range))
    rot_z = math.radians(random.uniform(-rot_z_range, rot_z_range))

    # Apply rotation relative to camera orientation
    # First get the "look at camera" rotation, then add random variations
    # Direction vector from board to camera
    direction = cam_matrix.translation - world_pos
    # Make board's +Z (front face) point toward camera, with Y as up
    rot_quat = direction.to_track_quat('Z', 'Y')

    # Add random rotation offsets
    random_euler = Euler((rot_x, rot_y, rot_z), 'XYZ')
    final_rotation = rot_quat @ random_euler.to_quaternion()

    board_obj.rotation_mode = 'QUATERNION'
    board_obj.rotation_quaternion = final_rotation
    board_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)


def generate_animation(camera_name, board_name, n_keyframes, frames_between,
                       z_min, z_max, rotation_ranges, view_coverage, seed=None):
    """
    Generate the complete animation with random board poses.

    Args:
        camera_name: Name of the camera object
        board_name: Name of the board object
        n_keyframes: Number of keyframe poses
        frames_between: Number of frames to interpolate between keyframes
        z_min, z_max: Positive distance range in front of camera (in Blender units)
        rotation_ranges: Tuple of (x_range, y_range, z_range) in degrees
        view_coverage: Tuple (min, max) for view coverage fraction
        seed: Random seed for reproducibility
    """
    # Set random seed
    if seed is not None:
        random.seed(seed)

    # Get objects
    try:
        camera = bpy.data.objects[camera_name]
        board = bpy.data.objects[board_name]
    except KeyError as e:
        print(f"Error: Object not found - {e}")
        print(f"Available objects: {list(bpy.data.objects.keys())}")
        return False

    # Verify camera type
    if camera.type != 'CAMERA':
        print(f"Error: '{camera_name}' is not a camera object")
        return False

    print(f"\nGenerating animation with {n_keyframes} keyframes...")
    print(f"Distance from camera: [{z_min}, {z_max}] Blender units")
    print(f"Rotation ranges: X=±{rotation_ranges[0]}°, Y=±{rotation_ranges[1]}°, Z=±{rotation_ranges[2]}°")
    print(f"View coverage: {view_coverage[0]*100:.0f}%-{view_coverage[1]*100:.0f}%")
    print(f"Frames between keyframes: {frames_between}\n")

    # Clear existing animation data for the board
    if board.animation_data:
        board.animation_data_clear()

    # Generate keyframes
    for i in range(n_keyframes):
        # Calculate frame number
        frame_num = 1 + i * frames_between

        # Random Z distance
        z_distance = random.uniform(z_min, z_max)

        # Set the pose
        set_board_pose(board, camera, z_distance, view_coverage, rotation_ranges, frame_num)

        print(f"Keyframe {i+1}/{n_keyframes} set at frame {frame_num} (distance={z_distance:.2f})")

    # Set the scene's frame range
    total_frames = 1 + (n_keyframes - 1) * frames_between
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_frames

    # Set interpolation to smooth (Bezier)
    for fcurve in board.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'BEZIER'
            keyframe.handle_left_type = 'AUTO_CLAMPED'
            keyframe.handle_right_type = 'AUTO_CLAMPED'

    print(f"\n✓ Animation generated successfully!")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / bpy.context.scene.render.fps:.2f}s at {bpy.context.scene.render.fps} FPS")

    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    success = generate_animation(
        camera_name=CAMERA_NAME,
        board_name=BOARD_NAME,
        n_keyframes=N_KEYFRAMES,
        frames_between=FRAMES_BETWEEN,
        z_min=Z_MIN,
        z_max=Z_MAX,
        rotation_ranges=(ROTATION_X_RANGE, ROTATION_Y_RANGE, ROTATION_Z_RANGE),
        view_coverage=VIEW_COVERAGE,
        seed=RANDOM_SEED
    )

    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Preview: Press SPACE or use the timeline to play animation")
        print("2. Adjust: Modify parameters at the top of this script and re-run")
        print("3. Render: Go to Render > Render Animation (or Ctrl+F12)")
        print("="*60)
