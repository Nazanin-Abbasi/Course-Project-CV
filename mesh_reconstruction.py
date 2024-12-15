import numpy as np
import open3d as o3d
import trimesh

def load_point_cloud(input_npy_path):
    # Loading point clouds
    data = np.load(input_npy_path)
    points = data[:, :3]
    colors = data[:, 3:6]

    if np.max(colors) > 1.0:
        colors = colors / 255.0

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

def estimate_normals(point_cloud):
    # Normal estimation
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
    point_cloud.orient_normals_consistent_tangent_plane(10)

def reconstruct_mesh(point_cloud, depth=9):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)

    # Remove low-density vertices
    low_density_vertices = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(low_density_vertices)

    return mesh

def transfer_colors(mesh, point_cloud, k=1):
    # Build KDTree
    point_cloud_colors = np.asarray(point_cloud.colors)
    point_cloud_tree = o3d.geometry.KDTreeFlann(point_cloud)

    # Initialize mesh vertex colors
    mesh_colors = np.zeros((len(mesh.vertices), 3))

    for i, v in enumerate(mesh.vertices):
        # Search for k nearest neighbors
        _, idx, dist = point_cloud_tree.search_knn_vector_3d(v, k)

        # Avoid division by zero by setting a minimum distance threshold
        dist = np.maximum(dist, 1e-6)

        # Compute weights inversely proportional to distances
        weights = 1.0 / dist
        weights /= np.sum(weights)

        # Compute the weighted average color
        weighted_color = np.sum(point_cloud_colors[idx] * weights[:, np.newaxis], axis=0)
        mesh_colors[i] = weighted_color

    # Assign colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)



def post_process_mesh(mesh, smoothing_iterations=50, smoothing_lambda=0.5, target_triangle_count=500000):
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

        # Remove problematic geometry
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    # Smooth the mesh
    mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=smoothing_iterations, lambda_filter=smoothing_lambda
    )
    mesh.compute_vertex_normals()

    # Simplify the mesh
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangle_count)
    mesh.compute_vertex_normals()

    # Cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    return mesh


def main():
    input_path = r"C:\Users\1080\Desktop\CV_Projects\frog_point_cloud_dec_14.npy"
    output_mesh_path = r"C:\Users\1080\Desktop\CV_Projects\frog_out_put.ply"
    output_processed_mesh_path = r"C:\Users\1080\Desktop\CV_Projects\frog_out_put_processed.ply"

    # Load and preprocess point cloud
    point_cloud = load_point_cloud(input_path)
    o3d.visualization.draw_geometries([point_cloud],
                                      window_name="Point Cloud Visualization",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50,
                                      point_show_normal=False)
    estimate_normals(point_cloud)

    # Mesh reconstruction
    mesh = reconstruct_mesh(point_cloud)
    transfer_colors(mesh, point_cloud)


    # Save initial mesh
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Initial colored mesh saved to:", output_mesh_path)

    # Post-process the mesh
    processed_mesh = post_process_mesh(mesh)

    # Save the post-processed mesh
    o3d.io.write_triangle_mesh(output_processed_mesh_path, processed_mesh)
    print("Post-processed mesh saved to:", output_processed_mesh_path)

if __name__ == "__main__":
    main()

