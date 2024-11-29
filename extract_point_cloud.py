import numpy as np


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


# Example usage
np.random.seed(42)
depth_frame = np.random.rand(10, 10) * 10  # Example 10x10 "depth" frame
point_cloud = frame_to_point_cloud(depth_frame, scale=1)

# Print the result
print("Point Cloud Shape:", point_cloud.shape)  # Should be (100, 3)
print("Sample Points:\n", point_cloud)  # Display the first 5 points
