import numpy as np
from solve_rt import apply_transformation


def pixel_to_camera_coords(point_cloud, height, width):
    # horizontal FOV
    f_x = width/1.1149825
    # vertical FOV
    f_y = height/0.83623225
    c_x = width/2
    c_y = height/2

    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    K = np.linalg.inv(K)

    d = np.array(point_cloud[:, 2])
    point_cloud[:, 2] = 1

    point_cloud[:, :3] = apply_transformation(point_cloud[:, :3], K, np.zeros(3))
    point_cloud[:, 0] *= d
    point_cloud[:, 1] *= d
    point_cloud[:, 2] = d

    return point_cloud
