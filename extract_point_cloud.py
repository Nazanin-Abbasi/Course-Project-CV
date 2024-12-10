import cv2
import numpy as np

from read_png import read_png_to_np
from solve_rt import apply_transformation


def frame_to_point_cloud(frame, scale=1.0):
    """
    Convert an n x m numpy array (e.g., an image or depth map) into a 3D point cloud.

    Parameters:
    - frame (np.ndarray): An n x m numpy array representing the frame (e.g., depth or intensity values).
    - scale (float): A scaling factor for the coordinates.

    Returns:
    - point_cloud (np.ndarray): An array of shape (n*m, 3) representing the [x, y, z] coordinates of the points.
    """
    n, m = frame.shape
    x_coords, y_coords = np.meshgrid(np.arange(m), np.arange(n))

    # Flatten the arrays and scale
    x_flat = x_coords.flatten() * scale
    y_flat = y_coords.flatten() * scale
    z_flat = frame.flatten() * scale

    # Combine into a single point cloud
    point_cloud = np.vstack((x_flat, y_flat, z_flat)).T
    return point_cloud


def remove_points_below_z(point_cloud, z_threshold=30):
    """
    Remove points from a point cloud where the z-coordinate is less than a given threshold.

    Parameters:
    - point_cloud (np.ndarray): Point cloud of shape (n, 3) with (x, y, z) coordinates.
    - z_threshold (float): Threshold for the z-coordinate. Points with z < threshold will be removed.

    Returns:
    - filtered_point_cloud (np.ndarray): Point cloud with points removed where z < threshold.
    """
    # Filter the point cloud based on the z-coordinate threshold
    filtered_point_cloud = point_cloud[point_cloud[:, 2] >= z_threshold]
    return filtered_point_cloud


def remove_points_above_z(point_cloud, z_threshold=10000):
    filtered_point_cloud = point_cloud[point_cloud[:, 2] <= z_threshold]
    return filtered_point_cloud


def frame_to_point_cloud_color(depth_frame, rgb_frame, scale=1.0):
    if depth_frame.shape[:2] != rgb_frame.shape[:2]:
        print("Resizing RGB frame to match depth frame dimensions.")
        rgb_frame = cv2.resize(rgb_frame, (depth_frame.shape[1], depth_frame.shape[0]))
    n, m = depth_frame.shape
    x_coords, y_coords = np.meshgrid(np.arange(m), np.arange(n))
    x_flat = x_coords.flatten() * scale
    y_flat = y_coords.flatten() * scale
    z_flat = depth_frame.flatten() * scale
    rgb_flat = rgb_frame.reshape(-1, 3) / 255.0  # Normalize RGB to [0, 1]
    point_cloud = np.hstack((np.vstack((x_flat, y_flat, z_flat)).T, rgb_flat))
    return point_cloud


def frame_to_point_cloud_color_np(depth_frame, rgb_frame, scale=1.0):
    n, m = depth_frame.shape
    x_coords, y_coords = np.meshgrid(np.arange(m), np.arange(n))
    x_flat = x_coords.flatten() * scale
    y_flat = y_coords.flatten() * scale
    z_flat = depth_frame.flatten() * scale
    rgb_flat = rgb_frame.reshape(-1, 3) / 255.0  # Normalize RGB to [0, 1]
    point_cloud = np.hstack((np.vstack((x_flat, y_flat, z_flat)).T, rgb_flat))
    return point_cloud


def normalize_point_cloud_by_dimension(points):
    """
    Normalize each dimension (x, y, z) of a 3D point cloud independently.

    Parameters:
        points: np.ndarray of shape (n, 3) - Point cloud with x, y, z coordinates.

    Returns:
        normalized_points: np.ndarray of shape (n, 3) - Normalized point cloud.
        min_vals: np.ndarray of shape (1, 3) - Minimum values for x, y, z (for unnormalization).
        max_vals: np.ndarray of shape (1, 3) - Maximum values for x, y, z (for unnormalization).
    """
    # Compute min and max for each dimension
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Avoid division by zero for degenerate ranges
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    # Normalize each dimension separately
    normalized_points = (points - min_vals) / ranges
    # print(min_vals, max_vals)
    return normalized_points


def normalize_point_cloud_by_dimension_with_rgb(points_with_rgb):
    """
    Normalize each dimension (x, y, z) of a 3D point cloud independently, ignoring RGB data.

    Parameters:
        points_with_rgb: np.ndarray of shape (n, 6) - Point cloud with x, y, z coordinates and RGB values.

    Returns:
        normalized_points_with_rgb: np.ndarray of shape (n, 6) - Normalized point cloud with RGB values retained.
        min_vals: np.ndarray of shape (1, 3) - Minimum values for x, y, z (for unnormalization).
        max_vals: np.ndarray of shape (1, 3) - Maximum values for x, y, z (for unnormalization).
    """
    # Extract the 3D points (first three columns)
    points = points_with_rgb[:, :3]

    # Compute min and max for each dimension
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Avoid division by zero for degenerate ranges
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    # Normalize each dimension separately
    normalized_points = (points - min_vals) / ranges

    # Combine normalized points with original RGB values
    normalized_points_with_rgb = np.hstack((normalized_points, points_with_rgb[:, 3:]))

    return normalized_points_with_rgb


def pixel_to_camera_coords(point_cloud):
    f_x = 500
    f_y = 500
    c_x = 80
    c_y = 60

    # Create the intrinsic matrix K
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    K = np.linalg.inv(K)
    # Convert to homogenous
    d = point_cloud[:, 2]
    # print(d)
    # point_cloud[:, 0] = point_cloud[:, 0] / d
    # point_cloud[:, 1] = point_cloud[:, 1] / d
    # point_cloud[:, 2] = point_cloud[:, 2] / d
    # Apply K
    point_cloud[:, :3] = apply_transformation(point_cloud[:, :3], K, np.zeros(3))
    point_cloud[:, 0] /= d
    point_cloud[:, 1] /= d
    # point_cloud[:, 2] = d

    # print(point_cloud[0])
    return point_cloud
