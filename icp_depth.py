import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from extract_point_cloud import (
    frame_to_point_cloud,
    frame_to_point_cloud_color,
    frame_to_point_cloud_color_np,
    normalize_point_cloud_by_dimension,
    normalize_point_cloud_by_dimension_with_rgb,
    pixel_to_camera_coords,
    remove_points_above_z,
    remove_points_below_z,
)
from find_closest_point import find_closest_point, find_closest_point_color
from read_png import read_png_to_np
from solve_rt import apply_transformation, compute_transformation
from visualize import downsample_point_cloud_percentage, visualize_3d_scatter

np.set_printoptions(precision=15)
filename = "Data/new_batch/160middleclose_depth.npy"
color_filename = "Data/new_batch/160middleclose_color.npy"

depth_frames = np.load(filename)
depth_frames = [depth_frames[i, 10:78, 50:120] for i in range(depth_frames.shape[0])]

color_frames = np.load(color_filename)
color_frames = [color_frames[i, 10:78, 50:120] for i in range(color_frames.shape[0])]

# point_clouds = [
#     frame_to_point_cloud(
#         depth_frame,
#     )
#     for depth_frame in depth_frames
# ]

point_clouds = [
    frame_to_point_cloud_color_np(depth_frame, color_frame)
    for depth_frame, color_frame in zip(depth_frames, color_frames)
]

point_clouds = [
    remove_points_below_z(point_cloud, z_threshold=3500) for point_cloud in point_clouds
]
point_clouds = [
    remove_points_above_z(point_cloud, z_threshold=6000) for point_cloud in point_clouds
]


point_clouds = [pixel_to_camera_coords(point_cloud) for point_cloud in point_clouds]

point_clouds = [
    normalize_point_cloud_by_dimension_with_rgb(point_cloud)
    for point_cloud in point_clouds
]

# point_clouds
# 14,35
point_clouds = [point_clouds[i] for i in range(14, 35)]
pairs = [
    (point_clouds[i][:, :3], point_clouds[i + 1][:, :3])
    for i in range(len(point_clouds) - 1)
]


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
        concencus_indices = np.where(d <= 0.01)[0]
        if len(concencus_indices) > max_concencus:
            max_concencus = len(concencus_indices)
            max_concencus_set = idx[concencus_indices]
            pc1_concencus_ind = concencus_indices

    return compute_transformation(pc1[pc1_concencus_ind], pc2[max_concencus_set])


def icp(pc1, pc2):
    idx, d = find_closest_point(pc2, pc1, option="ckd")
    reordered_reference = pc2[idx]
    return compute_transformation(pc1, reordered_reference)


outputs = []
c = 0
for pc1, pc2 in pairs:
    print(c)
    c += 1
    R_total = np.eye(3)  # 3x3 Identity matrix (no rotation)
    t_total = np.zeros(3)  # 3x1 Zero vector (no translation)
    p1 = pc1.copy()
    p2 = pc2.copy()
    for i in range(100):
        r, t = ransac_icp(p1, p2)
        # print(r, t)
        # r, t = icp(p, pc2)
        p1 = apply_transformation(p1, r, t)
        # Accumulate the transformation
        R_total = r @ R_total  # Multiply rotations (right-multiply)
        t_total = (
            r @ t_total
        ) + t  # Apply translation (rotate and then add translation)
    # Store the accumulated rotation and translation
    outputs.append((R_total, t_total))
    # visualize_3d_scatter(np.vstack((p1, p2)), percentage=100)

pc_d = {index: value for index, value in enumerate(point_clouds)}

for key in list(pc_d.keys()):
    for i in range(len(outputs)):
        if key <= i:
            r, t = outputs[i]
            pc_d[key][:, :3] = apply_transformation(pc_d[key][:, :3], r, t)


pcs = np.vstack(list(pc_d.values()))
print(pcs.shape)
# print(pcs.shape)
# np.save("frog_point_cloud_dec_9.npy", pcs)

point_cloud = pcs
visualize_3d_scatter(pcs, percentage=10)
