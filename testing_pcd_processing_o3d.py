import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud('point_clouds/kitti-frames_pointcloud.ply')
print([pcd])
print(type(pcd))
print(pcd.points)

points = np.asarray(pcd.points)
print(points)
print(points.shape)

o3d.visualization.draw_geometries(
    [pcd], zoom=0.6412
)

downpcd = pcd.voxel_down_sample(voxel_size=0.05)

o3d.visualization.draw_geometries(
    [downpcd],
    zoom=0.6412,
)

downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)

o3d.visualization.draw_geometries(
    [downpcd],
    point_show_normal=True
)

# demo_crop_data = o3d.data.DemoCropPointCloud()
# pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.7,
#                                   front=[0.5439, -0.2333, -0.8060],
#                                   lookat=[2.4615, 2.1331, 1.338],
#                                   up=[-0.1781, -0.9708, 0.1608])
# vol = o3d.visualization.read_selection_polygon_volume(
#     demo_crop_data.cropped_json_path)
# chair = vol.crop_point_cloud(pcd)
# o3d.visualization.draw_geometries([chair],
#                                   zoom=0.7,
#                                   front=[0.5439, -0.2333, -0.8060],
#                                   lookat=[2.4615, 2.1331, 1.338],
#                                   up=[-0.1781, -0.9708, 0.1608])

# aabb = chair.get_axis_aligned_bounding_box()
# aabb.color = (1, 0, 0)
# obb = chair.get_oriented_bounding_box()
# obb.color = (0, 1, 0)
# o3d.visualization.draw_geometries([chair, aabb, obb],
#                                   zoom=0.7,
#                                   front=[0.5439, -0.2333, -0.8060],
#                                   lookat=[2.4615, 2.1331, 1.338],
#                                   up=[-0.1781, -0.9708, 0.1608])

# bunny = o3d.data.BunnyMesh()
# mesh = o3d.io.read_triangle_mesh(bunny.path)
# mesh.compute_vertex_normals()

# pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
# hull, _ = pcl.compute_convex_hull()
# hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
# hull_ls.paint_uniform_color((1, 0, 0))
# o3d.visualization.draw_geometries([pcl, hull_ls])

# ply_point_cloud = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.455,
                                  )
