import numpy as np


def find_closest_point(reference_point_cloud, other_point_cloud):
    """
    Find correspondences between two point clouds by finding the closest point
    in reference_point_cloud for each point in other_point_cloud using KDTree or linalg.norm

    Parameters:
    - reference_point_cloud (np.ndarray): Source point cloud of shape (n1, 3).
    - other_point_cloud (np.ndarray): Target point cloud of shape (n2, 3).

    Returns:
    - indices (np.ndarray): Array of indices into reference_point_cloud of the closest points for each point in other_point_cloud.
    - distances (np.ndarray): Array of distances to the closest points.
    """
    # Uncomment if using KDTree for fast nearest neighbor search
    # tree = cKDTree(reference_point_cloud)
    # distances, indices = tree.query(other_point_cloud)

    # For manual implementation, we'll use a brute force method
    indices = []
    distances = []

    for p2 in other_point_cloud:
        # Compute distances from p1 to all points in point_cloud2
        dist = np.linalg.norm(reference_point_cloud - p2, axis=1)  # Euclidean distance
        closest_idx = np.argmin(dist)  # Find the index of the minimum distance
        indices.append(closest_idx)
        distances.append(dist[closest_idx])

    return np.array(indices), np.array(distances)
