import time
import numpy as np
from extract_point_cloud import frame_to_point_cloud, remove_points_below_z
from find_closest_point import find_closest_point
from read_png import read_png_to_np
from solve_rt import apply_transformation, compute_transformation
from visualize import visualize_3d_scatter, visualize_3d_scatter_with_colors

np.set_printoptions(precision=15)

filename = "ukulele2_frames/cropped_frame_71.png"
filename2 = "ukulele2_frames/cropped_frame_94.png"
filename3 = "ukulele2_frames/cropped_frame_109.png"
filename4 = "ukulele2_frames/cropped_frame_121.png"

depth_frame = read_png_to_np(filename)
depth_frame2 = read_png_to_np(filename2)
depth_frame3 = read_png_to_np(filename3)
depth_frame4 = read_png_to_np(filename4)

point_cloud = frame_to_point_cloud(depth_frame, scale=1)
point_cloud = remove_points_below_z(point_cloud)
point_cloud2 = frame_to_point_cloud(depth_frame2, scale=1)
point_cloud2 = remove_points_below_z(point_cloud2)
point_cloud3 = frame_to_point_cloud(depth_frame3, scale=1)
point_cloud3 = remove_points_below_z(point_cloud3)
point_cloud4 = frame_to_point_cloud(depth_frame4, scale=1)
point_cloud4 = remove_points_below_z(point_cloud4)

pcs = np.vstack((point_cloud, point_cloud2, point_cloud3, point_cloud4))
visualize_3d_scatter(pcs, percentage=25)

pairs = [
    (point_cloud, point_cloud2),
    (point_cloud2, point_cloud3),
    (point_cloud3, point_cloud4),
]

outputs = []
for pc1, pc2 in pairs:
    R_total = np.eye(3)  # 3x3 Identity matrix (no rotation)
    t_total = np.zeros(3)  # 3x1 Zero vector (no translation)
    for i in range(100):
        idx, d = find_closest_point(pc2, pc1, option="ckd")
        reordered_reference = pc2[idx]
        r, t = compute_transformation(pc1, reordered_reference)
        pc1 = apply_transformation(pc1, r, t)

        # Accumulate the transformation
        R_total = r @ R_total  # Multiply rotations (right-multiply)
        t_total = t_total + t  # Apply translation (rotate and then add translation)

    # Store the accumulated rotation and translation
    outputs.append((R_total, t_total))

pc_d = {
    0: point_cloud,
    1: point_cloud2,
    2: point_cloud3,
    3: point_cloud4,
}

for key, value in pc_d.items():
    for i in range(len(outputs)):
        if key <= i:
            r, t = outputs[i]
            pc_d[key] = apply_transformation(pc_d[key], r, t)


pcs = np.vstack(list(pc_d.values()))
visualize_3d_scatter(pcs, percentage=25)
# visualize_3d_scatter_with_colors(list(pc_d.values()), percentage=25)
