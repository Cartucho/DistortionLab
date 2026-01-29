"""
Blender script to export camera intrinsic parameters.

Run this script in Blender to export camera intrinsics to a YAML file
that can be used by apply_distortion.py.

Usage in Blender:
    1. Open your .blend file with the camera you want to export
    2. Run this script from the Scripting tab, or:
       blender --background your_scene.blend --python blender_export_intrinsics.py
"""

import bpy
import os


def get_scene_resolution(scene):
    """Get render resolution considering percentage scale."""
    res_x = scene.render.resolution_x * scene.render.resolution_percentage / 100
    res_y = scene.render.resolution_y * scene.render.resolution_percentage / 100
    return int(res_x), int(res_y)


def get_sensor_size(sensor_fit, sensor_width, sensor_height):
    """Get the effective sensor size based on sensor fit mode."""
    if sensor_fit == "VERTICAL":
        return sensor_height
    return sensor_width


def get_sensor_fit(sensor_fit, size_x, size_y):
    """Determine effective sensor fit mode."""
    if sensor_fit == "AUTO":
        if size_x >= size_y:
            return "HORIZONTAL"
        else:
            return "VERTICAL"
    return sensor_fit


def get_camera_intrinsics(scene, camera=None):
    """
    Get intrinsic camera parameters: focal length and principal point.

    Based on: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera

    Returns:
        dict with fx, fy, cx, cy, res_x, res_y
    """
    if camera is None:
        camera = scene.camera

    if camera is None:
        raise ValueError("No camera found in scene")

    cam_data = camera.data

    # Get resolution
    res_x, res_y = get_scene_resolution(scene)

    # Get sensor parameters
    focal_length_mm = cam_data.lens
    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height

    sensor_size_mm = get_sensor_size(
        cam_data.sensor_fit, sensor_width, sensor_height
    )

    sensor_fit = get_sensor_fit(
        cam_data.sensor_fit,
        scene.render.pixel_aspect_x * res_x,
        scene.render.pixel_aspect_y * res_y,
    )

    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    if sensor_fit == "HORIZONTAL":
        view_fac_in_px = res_x
    else:
        view_fac_in_px = pixel_aspect_ratio * res_y

    # Calculate focal length in pixels
    pixel_size_mm_per_px = (sensor_size_mm / focal_length_mm) / view_fac_in_px
    f_x = 1.0 / pixel_size_mm_per_px
    f_y = (1.0 / pixel_size_mm_per_px) / pixel_aspect_ratio

    # Calculate principal point (accounting for lens shift)
    c_x = (res_x - 1) / 2.0 - cam_data.shift_x * view_fac_in_px
    c_y = (res_y - 1) / 2.0 + (cam_data.shift_y * view_fac_in_px) / pixel_aspect_ratio

    return {
        "fx": f_x,
        "fy": f_y,
        "cx": c_x,
        "cy": c_y,
        "res_x": res_x,
        "res_y": res_y,
        "focal_length_mm": focal_length_mm,
        "sensor_width_mm": sensor_width,
        "sensor_height_mm": sensor_height,
        "sensor_fit": cam_data.sensor_fit,
    }


def export_intrinsics(output_path=None):
    """
    Export camera intrinsics to a YAML file.

    Args:
        output_path: Path to save YAML file. If None, saves next to .blend file.
    """
    scene = bpy.context.scene

    # Get intrinsics
    intrinsics = get_camera_intrinsics(scene)

    # Determine output path
    if output_path is None:
        blend_path = bpy.data.filepath
        if blend_path:
            output_path = os.path.splitext(blend_path)[0] + "_intrinsics.yaml"
        else:
            output_path = "camera_intrinsics.yaml"

    # Write YAML manually (no external dependencies in Blender)
    yaml_content = f"""# Camera intrinsics exported from Blender
# Scene: {scene.name}
# Camera: {scene.camera.name if scene.camera else 'None'}

# Resolution
res_x: {intrinsics['res_x']}
res_y: {intrinsics['res_y']}

# Focal length in pixels
fx: {intrinsics['fx']:.6f}
fy: {intrinsics['fy']:.6f}

# Principal point in pixels
cx: {intrinsics['cx']:.6f}
cy: {intrinsics['cy']:.6f}

# Original Blender parameters (for reference)
focal_length_mm: {intrinsics['focal_length_mm']:.2f}
sensor_width_mm: {intrinsics['sensor_width_mm']:.2f}
sensor_height_mm: {intrinsics['sensor_height_mm']:.2f}
sensor_fit: {intrinsics['sensor_fit']}
"""

    with open(output_path, 'w') as f:
        f.write(yaml_content)

    print(f"Camera intrinsics exported to: {output_path}")
    print(f"  Resolution: {intrinsics['res_x']}x{intrinsics['res_y']}")
    print(f"  Focal length: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f} pixels")
    print(f"  Principal point: cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")

    return output_path


if __name__ == "__main__":
    export_intrinsics()
