import numpy as np
from scipy.spatial import cKDTree

np.set_printoptions(precision=15)


def find_closest_point(reference_point_cloud, other_point_cloud, option="np", exclude_first=False):
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
    if option == "ckd":
        # Uncomment if using KDTree for fast nearest neighbor search
        tree = cKDTree(reference_point_cloud)
        if exclude_first:
            distances, indices = tree.query(other_point_cloud, k=2)
            distances = distances[:, 1]
            indices = indices[:, 1]
        else:
            distances, indices = tree.query(other_point_cloud)
        return np.array(indices), np.array(distances)

    elif option == "np":
        # For manual implementation, we'll use a brute force method
        indices = []
        distances = []

        for p2 in other_point_cloud:
            # Compute distances from p1 to all points in point_cloud2
            dist = np.linalg.norm(
                reference_point_cloud - p2, axis=1
            )  # Euclidean distance
            closest_idx = np.argmin(dist)  # Find the index of the minimum distance
            indices.append(closest_idx)
            distances.append(dist[closest_idx])

        return np.array(indices), np.array(distances)
    return None


def find_closest_point_color(
    reference_point_cloud, other_point_cloud, weight_rgb=0.5, option="ckd"
):
    if option == "ckd":
        combined_reference = np.hstack(
            (reference_point_cloud[:, :3], weight_rgb * reference_point_cloud[:, 3:])
        )
        combined_target = np.hstack(
            (other_point_cloud[:, :3], weight_rgb * other_point_cloud[:, 3:])
        )
        tree = cKDTree(combined_reference)
        distances, indices = tree.query(combined_target)
        return indices, distances
    elif option == "np":
        combined_reference = np.hstack(
            (reference_point_cloud[:, :3], weight_rgb * reference_point_cloud[:, 3:])
        )
        combined_target = np.hstack(
            (other_point_cloud[:, :3], weight_rgb * other_point_cloud[:, 3:])
        )
        indices = []
        distances = []
        for target_point in combined_target:
            dist = np.linalg.norm(combined_reference - target_point, axis=1)
            closest_idx = np.argmin(dist)
            indices.append(closest_idx)
            distances.append(dist[closest_idx])
        return np.array(indices), np.array(distances)
    return None
