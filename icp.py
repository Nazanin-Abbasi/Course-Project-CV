import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from extract_point_cloud import (
    frame_to_point_cloud,
    frame_to_point_cloud_color,
    frame_to_point_cloud_color_np,
    remove_points_above_z,
    remove_points_below_z,
)
from find_closest_point import find_closest_point, find_closest_point_color
from read_png import read_png_to_np
from solve_rt import apply_transformation, compute_transformation
from visualize import downsample_point_cloud_percentage, visualize_3d_scatter

np.set_printoptions(precision=15)
# filenames = [
#     f"cropped_ukulele_no_hole/cropped_frame_{i}.png"
#     for i in [71, 94, 109, 121, 131, 141, 150, 159, 169, 177, 184]
# ]
# color_files = [
#     f"color_ukulele/color_frame_{i}.png"
#     for i in [71, 94, 109, 121, 131, 141, 150, 159, 169, 177, 184]
# ]
# depth_frames = [read_png_to_np(filename) for filename in filenames]
filename = "Data/new_batch/160middleclose_depth.npy"
depth_frames = np.load(filename)
depth_frames = [depth_frames[i, 10:80, 50:120] for i in range(depth_frames.shape[0])]
# color_frames = [cv2.imread(name, cv2.IMREAD_COLOR) for name in color_files]
colorfilename = "Data/new_batch/160middleclose_color.npy"
color_frames = np.load(colorfilename)
color_frames = [color_frames[i, 10:80, 50:120, :] for i in range(color_frames.shape[0])]

point_clouds = [
    frame_to_point_cloud_color_np(depth_frame, color_frame)
    for depth_frame, color_frame in zip(depth_frames, color_frames)
]


# point_clouds = [
#     frame_to_point_cloud_color(depth_frame, color_frame)
#     for depth_frame, color_frame in zip(depth_frames, color_frames)
#

point_clouds = [
    remove_points_below_z(point_cloud, z_threshold=3500) for point_cloud in point_clouds
]
point_clouds = [
    remove_points_above_z(point_cloud, z_threshold=6250) for point_cloud in point_clouds
]


point_clouds = point_clouds[13:19]
# pcs = np.vstack(point_clouds)
# visualize_3d_scatter(pcs, percentage=100)
# assert False
# visualize_3d_scatter(point_clouds[0], percentage=10)
# assert False
pairs = [(point_clouds[i], point_clouds[i + 1]) for i in range(len(point_clouds) - 1)]


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


# def ransac_icp(pc1, pc2):
#     num_sample = 50
#     num_iterations = 100
#     max_concencus = 0
#     max_concencus_set = None

#     for iteration in range(num_iterations):
#         pc1_temp, pc2_temp = downsample_to_equal_points(pc1, pc2, num_sample)
#         idx, d = find_closest_point(pc2_temp, pc1_temp, option="ckd")
#         reordered_reference = pc2_temp[idx]
#         r, t = compute_transformation(pc1_temp, reordered_reference)
#         pc1_trans = apply_transformation(pc1, r, t)
#         idx, d = find_closest_point(pc2, pc1_trans, option="ckd")
#         concencus_indices = np.where(d < 0.6)[0]
#         if len(concencus_indices) > max_concencus:
#             max_concencus = len(concencus_indices)
#             max_concencus_set = idx[concencus_indices]
#             pc1_concencus_ind = concencus_indices

#     return compute_transformation(pc1[pc1_concencus_ind], pc2[max_concencus_set])


def ransac_icp_color(pc1, pc2):
    num_sample = 50
    num_iterations = 100
    max_concencus = 0
    max_concencus_set = None

    for iteration in range(num_iterations):
        pc1_temp, pc2_temp = downsample_to_equal_points(pc1, pc2, num_sample)
        idx, d = find_closest_point_color(
            pc2_temp, pc1_temp, weight_rgb=0.5, option="ckd"
        )
       
        reordered_reference = pc2_temp[idx]
        r, t = compute_transformation(pc1_temp[:, :3], reordered_reference[:, :3])
        pc1_trans = np.ndarray(pc1.shape)
        pc1_trans[:, :3] = apply_transformation(pc1[:, :3], r, t)
        pc1_trans[:, 3:] = pc1[:, 3:]
        idx, d = find_closest_point_color(pc2, pc1_trans, weight_rgb=0.5, option="ckd")
        concencus_indices = np.where(d < 0.6)[0]
        if len(concencus_indices) > max_concencus:
            max_concencus = len(concencus_indices)
            max_concencus_set = idx[concencus_indices]
            pc1_concencus_ind = concencus_indices

    return compute_transformation(
        pc1[pc1_concencus_ind][:, :3], pc2[max_concencus_set][:, :3]
    )


outputs = []
for pc1, pc2 in pairs:
    R_total = np.eye(3)  # 3x3 Identity matrix (no rotation)
    t_total = np.zeros(3)  # 3x1 Zero vector (no translation)
    for i in range(10):
        # r, t = ransac_icp(pc1, pc2)
        r, t = ransac_icp_color(pc1, pc2)
        pc1[:, :3] = apply_transformation(pc1[:, :3], r, t)
        # Accumulate the transformation
        R_total = r @ R_total  # Multiply rotations (right-multiply)
        t_total = (
            r @ t_total
        ) + t  # Apply translation (rotate and then add translation)
    # Store the accumulated rotation and translation
    outputs.append((R_total, t_total))


# filenames = [
#     f"ukulele2_frames/frame_{i}.png"
#     for i in [71, 94, 109, 121, 131, 141, 150, 159, 169, 177, 184]
# ]
# color_files = [
#     f"color_ukulele/color_frame_{i}.png"
#     for i in [71, 94, 109, 121, 131, 141, 150, 159, 169, 177, 184]
# ]
# depth_frames = [read_png_to_np(filename) for filename in filenames]
# color_frames = [cv2.imread(name, cv2.IMREAD_COLOR) for name in color_files]

# depth_masks = [np.where(arr.flatten() == 0) for arr in depth_frames]
# point_clouds = [frame_to_point_cloud(depth_frame) for depth_frame in depth_frames]
# point_clouds = [
#     frame_to_point_cloud_color(depth_frame, color_frame)
#     for depth_frame, color_frame in zip(depth_frames, color_frames)
# ]

# point_clouds = [remove_points_below_z(point_cloud) for point_cloud in point_clouds]

pc_d = {index: value for index, value in enumerate(point_clouds)}

for key in list(pc_d.keys()):
    for i in range(len(outputs)):
        if key <= i:
            r, t = outputs[i]
            pc_d[key][:, :3] = apply_transformation(pc_d[key][:, :3], r, t)


pcs = np.vstack(list(pc_d.values()))

# print(pcs.shape)
# np.save("ukulele_cloud_using_color.npy", pcs)

point_cloud = pcs
visualize_3d_scatter(pcs, percentage=100)
assert False

point_cloud = downsample_point_cloud_percentage(point_cloud, percentage=10)
# Separate into coordinates and color components
x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
r, g, b = point_cloud[:, 3], point_cloud[:, 4], point_cloud[:, 5]

# Normalize RGB values if not already in range [0, 1]
rgb_colors = np.clip(np.column_stack((r, g, b)), 0, 1)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(x, y, z, c=rgb_colors, marker="o", s=1)

# Set labels for clarity
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Show the plot
plt.show()
