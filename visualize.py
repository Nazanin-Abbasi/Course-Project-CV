import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def visualize_3d_scatter(point_cloud, percentage=100):
    """
    Visualize a 3D point cloud using Matplotlib's 3D scatter plot.

    Parameters:
    - point_cloud (np.ndarray): Point cloud of shape (n, 3) with (x, y, z) coordinates.

    Returns:
    - None
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    point_cloud = downsample_point_cloud_percentage(point_cloud, percentage)

    # Unpack the point cloud
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    # Scatter plot
    scatter = ax.scatter(x, y, z, c=z, cmap="viridis", s=1)
    fig.colorbar(scatter, ax=ax, label="Depth (Z)")

    # Labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud Visualization")

    plt.show()


def downsample_point_cloud_percentage(point_cloud, percentage):
    """
    Downsample a point cloud by a given percentage.

    Parameters:
    - point_cloud (np.ndarray): Point cloud of shape (n, 3).
    - percentage (float): Percentage of points to retain (0 < percentage <= 100).

    Returns:
    - downsampled_point_cloud (np.ndarray): Downsampled point cloud.
    """
    num_points = int(
        len(point_cloud) * percentage / 100
    )  # Calculate the number of points to sample
    indices = np.random.choice(
        point_cloud.shape[0], num_points, replace=False
    )  # Randomly sample indices
    downsampled_point_cloud = point_cloud[indices]  # Get the sampled points
    return downsampled_point_cloud


def visualize_3d_scatter_with_colors(point_clouds, percentage=100):
    """
    Visualize multiple 3D point clouds with different colors.

    Parameters:
    - point_clouds (list of np.ndarray): List of point clouds where each element is a point cloud (n, 3).

    Returns:
    - None
    """
    fig = go.Figure()

    # Define a list of colors for each point cloud
    colors = ["blue", "green", "red", "orange", "purple"]

    for i, point_cloud in enumerate(point_clouds):
        point_cloud = downsample_point_cloud_percentage(point_cloud, percentage)
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=4,
                    color=colors[i % len(colors)],  # Cycle through the colors
                    opacity=0.8,
                ),
                name=f"Point Cloud {i+1}",  # Add a label for the point cloud
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Point Clouds with Different Colors",
        showlegend=True,  # Show legend to identify different point clouds
    )

    fig.show()
