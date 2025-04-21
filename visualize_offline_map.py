import open3d as o3d

map_file = "output/slam_map_colored.ply"
# traj_file = "output/slam_trajectory.ply"
imu_file = "output/imu_trajectory.ply"

# Load the color-coded map
pc_map = o3d.io.read_point_cloud(map_file)

# Load and color the main (ground truth) trajectory
# pc_traj = o3d.io.read_point_cloud(traj_file)
# pc_traj.paint_uniform_color([1.0, 0.0, 0.0])  # Red

# Load and color the IMU trajectory
pc_imu = o3d.io.read_point_cloud(imu_file)
pc_imu.paint_uniform_color([1.0, 0.0, 0.0])  # Black

print("✅ Loaded colored SLAM map, ground truth and IMU trajectory")
o3d.visualization.draw_geometries([pc_map, pc_imu])
