import numpy as np

from find_closest_point import find_closest_point
from solve_rt import apply_transformation, compute_transformation
from visualize import (visualize_3d_scatter,
                       visualize_3d_scatter_groups,
                       visualize_3d_scatter_colorized)

np.set_printoptions(precision=15)

def downsample_to_equal_points(pc1, pc2, target_size=None):
    """
    Downsample two point clouds to the same number of points.

    Parameters:
    - pc1: np.ndarray, shape (n, 3), first point cloud
    - pc2: np.ndarray, shape (m, 3), second point cloud
    - target_size: int, optional, target number of points.
      If None, the smaller size between the two point clouds is used.

    Returns:
    - downsampled_pc1: np.ndarray, shape (k, 3)
    - downsampled_pc2: np.ndarray, shape (k, 3)
    """
    n, m = pc1.shape[0], pc2.shape[0]

    if target_size is None:
        target_size = min(n, m)

    def downsample_random(pc, target_size):
        if pc.shape[0] <= target_size:
            # If the point cloud already has fewer or equal points, just return it
            return pc
        # Random sampling
        indices = np.random.choice(pc.shape[0], target_size, replace=False)
        return pc[indices]

    downsampled_pc1 = downsample_random(pc1, target_size)
    downsampled_pc2 = downsample_random(pc2, target_size)

    return downsampled_pc1, downsampled_pc2


def ransac_icp(pc1, pc2):
    num_sample = 50
    num_iterations = 50
    max_concencus = 0
    max_concencus_set = None

    for iteration in range(num_iterations):
        pc1_temp, pc2_temp = downsample_to_equal_points(pc1, pc2, num_sample)
        idx, d = find_closest_point(pc2_temp, pc1_temp, option="ckd")
        reordered_reference = pc2_temp[idx]
        r, t = compute_transformation(pc1_temp, reordered_reference)
        pc1_trans = apply_transformation(pc1, r, t)
        idx, d = find_closest_point(pc2, pc1_trans, option="ckd")
        concencus_indices = np.where(d <= 10)[0]
        if len(concencus_indices) > max_concencus:
            max_concencus = len(concencus_indices)
            max_concencus_set = idx[concencus_indices]
            pc1_concencus_ind = concencus_indices

    return compute_transformation(pc1[pc1_concencus_ind], pc2[max_concencus_set])


def icp(pc1, pc2):
    idx, d = find_closest_point(pc2, pc1, option="ckd")
    reordered_reference = pc2[idx]
    return compute_transformation(pc1, reordered_reference)


def remove_furthest_points(p1, p2, percentage=10):
    idx1, d1 = find_closest_point(p2, p1, "ckd")
    idx2, d2 = find_closest_point(p1, p2, "ckd")
    top_n1 = max(1, int(len(d1) * percentage/100))
    top_n2 = max(1, int(len(d2) * percentage/100))

    indexes1 = np.argsort(d1)[:-top_n1]
    indexes2 = np.argsort(d2)[:-top_n2]
    return p1[indexes1, :], p2[indexes2, :]


def run_icp(point_clouds):
    print("Starting ICP")
    pairs = [
        (point_clouds[i][:, :3], point_clouds[i + 1][:, :3])
        for i in range(len(point_clouds) - 1)
    ]

    outputs = []
    c = 1
    for pc1, pc2 in pairs:
        print(f"{c}/{len(pairs)}")
        c += 1
        R_total = np.eye(3)
        t_total = np.zeros(3)
        first = True
        for mega_iter in range(3):
            p1 = pc1.copy()
            p2 = pc2.copy()
            p1 = apply_transformation(p1, R_total, t_total)
            if not first:
                p1, p2 = remove_furthest_points(p1, p2, mega_iter * 5)
            for iter in range(10):
                repeats = 4
                if first:
                    repeats = 10
                    first = False
                for i in range(repeats):
                    r, t = icp(p1, p2)
                    p1 = apply_transformation(p1, r, t)
                    R_total = r @ R_total
                    t_total = (r @ t_total) + t
                p1, p2 = remove_furthest_points(p1, p2, 2)
        outputs.append((R_total, t_total))

    pc_d = {index: value for index, value in enumerate(point_clouds)}

    for key in list(pc_d.keys()):
        for i in range(len(outputs)):
            if key <= i:
                r, t = outputs[i]
                pc_d[key][:, :3] = apply_transformation(pc_d[key][:, :3], r, t)

    return list(pc_d.values())
