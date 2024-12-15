import numpy as np
import open3d as o3d
import trimesh
import pymesh

def load_point_cloud(input_npy_path):
    """
    Load point cloud data from a .npy file and return a PointCloud object.

    Parameters
    ----------
    input_npy_path : str
        Path to the .npy file containing the point cloud data.

    Returns
    -------
    o3d.geometry.PointCloud
        The loaded point cloud.
    """
    data = np.load(input_npy_path)
    points = data[:, :3]
    colors = data[:, 3:6]

    if np.max(colors) > 1.0:
        colors = colors / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def estimate_normals(pcd):
    """
    Estimate normals for the given point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The input point cloud.

    Returns
    -------
    None
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)

def reconstruct_mesh(pcd, depth=10):
    """
    Perform Poisson surface reconstruction on the point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The input point cloud.
    depth : int, optional
        Depth parameter for Poisson reconstruction, by default 8.

    Returns
    -------
    o3d.geometry.TriangleMesh
        The reconstructed mesh.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh

def transfer_colors(mesh, pcd, k=50):
    """
    Transfer colors from a point cloud to a mesh using weighted averaging of the nearest neighbors.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The input mesh.
    pcd : o3d.geometry.PointCloud
        The input point cloud.
    k : int, optional
        Number of nearest neighbors to consider for weighted color transfer, by default 3.

    Returns
    -------
    None
    """
    # Preload point cloud colors and build KDTree
    pcd_colors = np.asarray(pcd.colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # Initialize mesh vertex colors
    mesh_colors = np.zeros((len(mesh.vertices), 3))

    # Iterate through each vertex in the mesh
    for i, v in enumerate(mesh.vertices):
        # Search for k nearest neighbors
        _, idx, dist = pcd_tree.search_knn_vector_3d(v, k)

        # Avoid division by zero by setting a minimum distance threshold
        dist = np.maximum(dist, 1e-6)

        # Compute weights inversely proportional to distances
        weights = 1.0 / dist
        weights /= np.sum(weights)  # Normalize weights

        # Compute the weighted average color
        weighted_color = np.sum(pcd_colors[idx] * weights[:, np.newaxis], axis=0)
        mesh_colors[i] = weighted_color

    # Assign colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

def post_process_mesh(mesh, remove_small_clusters_threshold=100, smoothing_iterations=20, smoothing_lambda=0.5, target_triangle_count=50000):
    """
    Perform various post-processing steps on a mesh.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The input mesh to be post-processed.
    remove_small_clusters_threshold : int, optional
        Threshold for removing small clusters, by default 100.
    smoothing_iterations : int, optional
        Number of smoothing iterations, by default 20.
    smoothing_lambda : float, optional
        Smoothing factor, by default 0.5.
    target_triangle_count : int, optional
        Target number of triangles after decimation, by default 50000.

    Returns
    -------
    o3d.geometry.TriangleMesh
        The post-processed mesh.
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=smoothing_iterations, lambda_filter=smoothing_lambda
    )
    mesh.compute_vertex_normals()

    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    large_clusters = np.where(np.array(cluster_n_triangles) > remove_small_clusters_threshold)[0]
    triangles_to_keep = [i for i, cidx in enumerate(triangle_clusters) if cidx in large_clusters]
    mesh = mesh.select_by_index(triangles_to_keep)

    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangle_count)
    mesh.compute_vertex_normals()

    return mesh

def remove_small_components(mesh, min_cluster_size=100):
    """
    Remove small disconnected components from the mesh.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The input mesh.
    min_cluster_size : int
        Minimum cluster size to retain.

    Returns
    -------
    o3d.geometry.TriangleMesh
        The cleaned mesh.
    """
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    # Filter clusters by size
    triangles_to_keep = [i for i, cidx in enumerate(triangle_clusters)
                         if cluster_n_triangles[cidx] > min_cluster_size]

    # Select remaining triangles
    mesh = mesh.select_by_index(triangles_to_keep)
    return mesh


def fill_mesh_holes_trimesh(mesh):
    """
    Fill holes in a mesh using Trimesh.
    """
    trimesh_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))

    # Fill holes
    trimesh_mesh.fill_holes()

    # Return the updated Open3D mesh
    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
        triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces)
    )

def main():
    input_npy_path = r"C:\Users\1080\Desktop\CV_Projects\frog_point_cloud_dec_14.npy"
    output_mesh_path = r"C:\Users\1080\Desktop\CV_Projects\frog_out_put.ply"
    output_processed_mesh_path = r"C:\Users\1080\Desktop\CV_Projects\frog_out_put_processed.ply"

    # Load and preprocess point cloud
    pcd = load_point_cloud(input_npy_path)
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Point Cloud Visualization",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50,
                                      point_show_normal=False)
    estimate_normals(pcd)

    # Mesh reconstruction
    mesh = reconstruct_mesh(pcd)
    mesh = fill_mesh_holes_trimesh(mesh)
    transfer_colors(mesh, pcd)
    # Fill holes in the mesh

    # Save initial mesh
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Initial colored mesh saved to:", output_mesh_path)

    # Post-process the mesh
    processed_mesh = post_process_mesh(mesh)
    processed_mesh=remove_small_components(processed_mesh)
    mesh2 = fill_mesh_holes_trimesh(processed_mesh)

    # Save the post-processed mesh
    o3d.io.write_triangle_mesh(output_processed_mesh_path, mesh2)
    print("Post-processed mesh saved to:", output_processed_mesh_path)

if __name__ == "__main__":
    main()

