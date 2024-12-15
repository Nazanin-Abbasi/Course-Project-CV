import numpy as np
from extract_point_cloud import (
    frame_to_point_cloud_color_np,
    clean_up_point_clouds
)
from pixel_to_camera_coords import pixel_to_camera_coords
from visualize import (visualize_3d_scatter,
                       visualize_3d_scatter_groups,
                       visualize_3d_scatter_colorized)
from icp_depth import run_icp
from remove_outliers import remove_outliers


if __name__ == "__main__":
    # HYPERPARAMETERS
    INPUT_DIRECTORY = "Data/new_batch/"
    OUTPUT_FILENAME = "frog_point_cloud_dec_14.npy"
    CROP = [[10, 78], [50, 120]]
    FRAMES = [14, 35, 1]
    DOWNSAMPLE_COLOR = 2
    MIN_DEPTH = 3500
    MAX_DEPTH = 6000
    REMOVE_PLANE = False

    # INPUT_DIRECTORY = "Data/lama/"
    # OUTPUT_FILENAME = "lama_point_cloud_dec_14.npy"
    # CROP = None
    # FRAMES = [80, 180, 4]
    # DOWNSAMPLE_COLOR = False
    # MIN_DEPTH = 300
    # MAX_DEPTH = 1000
    # REMOVE_PLANE = True

    # ===========================================
    filename = INPUT_DIRECTORY + "depth.npy"
    color_filename = INPUT_DIRECTORY + "color.npy"

    depth_frames = np.load(filename)
    height, width = depth_frames[0].shape
    if CROP is not None:
        depth_frames = [depth_frames[i, CROP[0][0]:CROP[0][1], CROP[1][0]:CROP[1][1]] for i in range(depth_frames.shape[0])]

    color_frames = np.load(color_filename)
    if DOWNSAMPLE_COLOR:
        color_frames = np.array([color_frames[i, ::DOWNSAMPLE_COLOR, ::DOWNSAMPLE_COLOR] for i in range(color_frames.shape[0])])
    if CROP is not None:
        color_frames = [color_frames[i, CROP[0][0]:CROP[0][1], CROP[1][0]:CROP[1][1]] for i in range(color_frames.shape[0])]

    point_clouds = [
        frame_to_point_cloud_color_np(depth_frame, color_frame, CROP)
        for depth_frame, color_frame in zip(depth_frames, color_frames)
    ]

    point_clouds = [point_clouds[i] for i in range(FRAMES[0], FRAMES[1], FRAMES[2])]
    point_clouds = [pixel_to_camera_coords(point_cloud, height, width) for point_cloud in point_clouds]

    point_clouds = clean_up_point_clouds(point_clouds, MIN_DEPTH, MAX_DEPTH, REMOVE_PLANE, depth_frames[0])

    point_clouds = run_icp(point_clouds)
    point_clouds = remove_outliers(point_clouds)

    # Visualize the data
    visualize_3d_scatter_groups(point_clouds, percentage=100)

    pc_stacked = np.vstack(point_clouds)
    visualize_3d_scatter_colorized(pc_stacked, percentage=100)

    visualize_3d_scatter(pc_stacked, percentage=100)

    np.save(OUTPUT_FILENAME, pc_stacked)
