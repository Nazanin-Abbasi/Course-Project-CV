import numpy as np
from matplotlib import pyplot as plt

point_cloud = np.load("frog_point_cloud_dec_9.npy")


# Extract x, y, z and RGB values
xyz = point_cloud[:, :3]
rgb = point_cloud[:, 3:]  # Normalize RGB to [0, 1] for Matplotlib

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=20)  # Use RGB as color

# Set labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
