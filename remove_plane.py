import numpy as np
import matplotlib.pyplot as plt
from pixel_to_camera_coords import pixel_to_camera_coords

class plane:
    def __init__(self, pos, orientation):
        self.position = pos
        self.orientation = orientation

    def check_over(self, point, distance=0):
        return distance <= (point - self.position) @ self.orientation


# Define the click event handler
clicked_coordinates = [(0, 0)]
def on_click(event):
    if event.inaxes:  # Ensure the click is within the image
        x, y = int(event.xdata), int(event.ydata)
        clicked_coordinates[0] = (x, y)
        # Disconnect the event after the first click
        plt.close()


def approximate_plane(points):
    """
    Fits a plane to a set of 3D points using SVD.

    Args:
        points (ndarray): N x 3 array of 3D points.

    Returns:
        normal (ndarray): Normal vector of the plane [a, b, c].
        d (float): The plane offset.
    """
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Center the points
    centered_points = points - centroid

    # Compute the SVD
    _, _, vh = np.linalg.svd(centered_points)

    # The normal vector to the plane is the last singular vector
    normal = vh[-1]
    if normal[2] > 0: normal = -normal

    return plane(centroid, normal)


def identify_plane(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.title("Click on a plane to remove it and anything under it")
    plt.show()
    if clicked_coordinates[0] == (0, 0):
        return None
    x, y = clicked_coordinates[0]
    points = np.array([[i, j, image[j, i], 0.0, 0.0, 0.0] for i in range(x-3, x+4) for j in range(y-3, y+4)])
    points = pixel_to_camera_coords(points, image.shape[0], image.shape[1])
    points = points[:, :3]

    return approximate_plane(points)


def remove_under_plane(plane, point_cloud):
    return np.array([point for point in point_cloud if plane.check_over(point[:3], 40)])
