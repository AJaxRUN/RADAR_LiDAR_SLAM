import open3d as o3d

map_file = "output/map_colored.ply"
imu_file = "output/imu_trajectory.ply"

pc_map = o3d.io.read_point_cloud(map_file)

pc_imu = o3d.io.read_point_cloud(imu_file)
pc_imu.paint_uniform_color([1.0, 0.0, 0.0])  # Black

print("Loaded colored map, ground truth and IMU trajectory")
o3d.visualization.draw_geometries([pc_map, pc_imu])
