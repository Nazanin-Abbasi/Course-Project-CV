import cv2
import numpy as np

from find_closest_point import find_closest_point

depth_ts = "Data/ukulele2/depth_timesteps.npy"
color_ts = "Data/ukulele2/color_timesteps.npy"
depth_frames = [71, 94, 109, 121, 131, 141, 150, 159, 169, 177, 184]
depth_ts = np.load(depth_ts)
color_ts = np.load(color_ts)


def find_closest_indices(arr1, arr2):
    """
    Find the index of the closest point in arr1 for each point in arr2.

    Parameters:
    - arr1: np.ndarray, shape (m,)
    - arr2: np.ndarray, shape (n,)

    Returns:
    - closest_indices: list of indices in arr1 corresponding to closest points
    """
    # Compute pairwise absolute differences
    distances = np.abs(arr1[:, None] - arr2[None, :])  # Shape (m, n)
    # Find the index of the minimum distance for each point in arr2
    closest_indices = np.argmin(distances, axis=0)
    return closest_indices


# x = find_closest_point(color_ts, depth_ts[depth_frames], option="ckd"),
x = find_closest_indices(color_ts, depth_ts[depth_frames])
# print(color_ts[121], depth_ts[121])
all_color = np.load("Data/ukulele2/color.npy")[x]
all_color = all_color[:, 70:240, :260, :]
# frame = depth_frames[current_frame, 70:240, 0:260]
for i in range(all_color.shape[0]):
    frame = all_color[i]
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame_normalized = frame_normalized.astype("uint8")
    cv2.imwrite(f"color_ukulele_full/color_frame_{depth_frames[i]}.png", frame_normalized)
