import numpy as np
from find_closest_point import find_closest_point

np.set_printoptions(precision=15)


def remove_outliers(point_clouds):
    print("Starting removal")
    average_closest_distance = np.mean(find_closest_point(point_clouds[0], point_clouds[0], "ckd", True)[1])
    for i in range(len(point_clouds)):
        print(f"{i+1}/{len(point_clouds)}")
        # find the distance for each point to each other point cloud
        # if all the distances are above 0.6* average distance remove it
        point_cloud = point_clouds[i]
        keep = np.full(point_cloud.shape[0], False)
        for j in range(len(point_clouds) - 1):
            index = ((-1) ** j * (j // 2 + 1) + i) % len(point_clouds)
            other_point_cloud = point_clouds[index]
            _, d = find_closest_point(other_point_cloud, point_cloud, "ckd")
            keep = keep | (d <= average_closest_distance * 0.6)
        point_clouds[i] = point_cloud[keep]
    return point_clouds
