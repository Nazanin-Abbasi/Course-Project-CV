import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_transformation(P, Q):

    # Step 1: Compute centroids
    c_P = np.mean(P, axis=0)
    c_Q = np.mean(Q, axis=0)

    # Step 2: Centralize points
    P_prime = P - c_P
    Q_prime = Q - c_Q

    # Step 3: Compute cross-covariance matrix
    H = P_prime.T @ Q_prime

    # Step 4: Perform SVD
    U, _, Vt = np.linalg.svd(H)

    # Step 5: Compute rotation matrix
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 6: Compute translation vector
    T = c_Q - R @ c_P

    return R, T


def apply_transformation(cloud, r, t):
    # Ensure t is a (1, 3) vector for broadcasting
    t = t.reshape(1, 3)
    # Apply rotation and translation
    transformed_point_cloud = np.dot(cloud, r.T) + t
    return transformed_point_cloud




# # Generate random 3D points
# np.random.seed(42)
# num_points = 10
# P = np.random.rand(num_points, 3) * 10  # Source point cloud

# # Create a rotation matrix (example rotation around Z-axis)
# theta = np.radians(180)  # Rotation angle in degrees
# R_actual = np.array(
#     [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
# )

# # Translation vector
# T_actual = np.array([5, -3, 2])

# # Apply rotation and translation to generate target point cloud
# Q = (R_actual @ P.T).T + T_actual

# # Compute the rotation and translation from P to Q
# R_computed, T_computed = compute_transformation(P, Q)

# # Apply the computed transformation to P
# P_transformed = (R_computed @ P.T).T + T_computed

# # Visualization
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# # Plot original source point cloud
# ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="r", label="Source (P)", alpha=0.6)

# # Plot target point cloud
# ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c="g", label="Target (Q)", alpha=0.6)

# # Plot transformed source point cloud
# ax.scatter(
#     P_transformed[:, 0],
#     P_transformed[:, 1],
#     P_transformed[:, 2],
#     c="b",
#     label="Transformed (P_transformed)",
#     alpha=0.6,
# )

# # Add labels and legend
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()
# ax.set_title("Visualization of Rotation and Translation")

# plt.show()
