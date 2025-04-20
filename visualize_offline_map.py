# visualize_offline_map.py
import open3d as o3d

map_file = "output/slam_map_colored.ply"
traj_file = "output/slam_trajectory.ply"

# Load the color-coded map
pc_map = o3d.io.read_point_cloud(map_file)

# Load and color the trajectory separately
pc_traj = o3d.io.read_point_cloud(traj_file)
pc_traj.paint_uniform_color([1.0, 0.0, 0.0])  # Red path

print("✅ Loaded colored SLAM map and trajectory")
o3d.visualization.draw_geometries([pc_map, pc_traj])
