import carla
import numpy as np
import threading
import time
import base64
from io import BytesIO
from PIL import Image
import os
import open3d as o3d
import torch
from pointnet_completion import PointNetCompletion

# Initialize the neural network model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNetCompletion(input_points=256, output_points=2048).to(device)
model.load_state_dict(torch.load("pointnet_completion.pth", weights_only=True))
model.eval()

imu_trajectory = []
camera_image = {'data': None}
pos_noise_std = 0.2
lidar_map_points = []
radar_map_points = []
nn_lidar_points = []
trajectory = []

def carla_slam_loop():
    global trajectory, lidar_map_points, radar_map_points, nn_lidar_points

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    if world.get_map().name != "Town02":
        print("Loading Town02...")
        client.load_world("Town02")
        time.sleep(5)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        weather = carla.WeatherParameters(
            fog_density=75, cloudiness=100.0, precipitation=0.0, sun_altitude_angle=10.0
        )
        world.set_weather(weather)
        print("Fog weather applied.")

    for light in world.get_actors().filter("traffic.traffic_light"):
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(9999)
        light.freeze(True)
    print("All traffic lights set to GREEN.")

    for actor in world.get_actors().filter("*vehicle*"):
        actor.destroy()
    for actor in world.get_actors().filter("*walker*"):
        actor.destroy()
    print("Cleared all other actors.")

    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn)
    vehicle.set_autopilot(True)

    lidar_tf = carla.Transform(carla.Location(x=0, z=2.5))
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "100.0")
    lidar_bp.set_attribute("channels", "128")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("points_per_second", "240000")
    lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)

    radar_bp = bp_lib.find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", "60")
    radar_bp.set_attribute("vertical_fov", "20")
    radar_bp.set_attribute("range", "100")
    radar1_tf = carla.Transform(carla.Location(x=1.0, y=0.0, z=2.0))
    radar1 = world.spawn_actor(radar_bp, radar1_tf, attach_to=vehicle)
    radar2_tf = carla.Transform(carla.Location(x=1.0, y=0.5, z=2.0))
    radar2 = world.spawn_actor(radar_bp, radar2_tf, attach_to=vehicle)
    radar3_tf = carla.Transform(carla.Location(x=1.0, y=-0.5, z=2.0))
    radar3 = world.spawn_actor(radar_bp, radar3_tf, attach_to=vehicle)

    spectator = world.get_spectator()
    def follow_vehicle():
        prev_location = None
        while True:
            transform = vehicle.get_transform()
            forward = transform.get_forward_vector()
            right = transform.get_right_vector()
            up = transform.get_up_vector()
            dx, dy, dz = -6, 0, 3
            target_loc = transform.location + forward * dx + right * dy + up * dz
            alpha = 0.1
            if prev_location is None:
                smoothed_loc = target_loc
            else:
                smoothed_loc = prev_location * (1 - alpha) + target_loc * alpha
            prev_location = smoothed_loc
            cam_rotation = carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(smoothed_loc, cam_rotation))
            time.sleep(0.05)

    threading.Thread(target=follow_vehicle, daemon=True).start()

    tm = client.get_trafficmanager()
    tm.ignore_lights_percentage(vehicle, 100.0)
    tm.ignore_signs_percentage(vehicle, 100.0)
    tm.ignore_vehicles_percentage(vehicle, 100.0)
    tm.set_global_distance_to_leading_vehicle(5.0)
    tm.set_synchronous_mode(False)

    latest_lidar = {'points': None}
    latest_radar = {'points': []}

    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.05")
    camera_bp.set_attribute("enable_postprocess_effects", "True")
    camera_tf = carla.Transform(carla.Location(x=1.5, z=3.0))
    camera = world.spawn_actor(camera_bp, camera_tf, attach_to=vehicle)

    # Sensor offsets relative to vehicle center
    lidar_offset = np.array([0.0, 0.0, 2.5])
    radar1_offset = np.array([1.0, 0.0, 2.0])
    radar2_offset = np.array([1.0, 0.5, 2.0])
    radar3_offset = np.array([1.0, -0.5, 2.0])

    def save_map_continuously():
        while True:
            try:
                if lidar_map_points or radar_map_points or nn_lidar_points:
                    combined = []
                    colors = []
                    if lidar_map_points:
                        lidar_np = np.array(lidar_map_points)
                        combined.append(lidar_np)
                        colors.append(np.tile([0, 0.6, 1.0], (len(lidar_np), 1)))
                    if radar_map_points:
                        radar_np = np.array(radar_map_points)
                        combined.append(radar_np)
                        colors.append(np.tile([0.2, 1.0, 0.0], (len(radar_np), 1)))
                    if nn_lidar_points:
                        nn_lidar_np = np.array(nn_lidar_points)
                        combined.append(nn_lidar_np)
                        colors.append(np.tile([0.5, 0.0, 0.5], (len(nn_lidar_np), 1)))
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(np.vstack(combined))
                    pc.colors = o3d.utility.Vector3dVector(np.vstack(colors))
                    o3d.io.write_point_cloud("output/slam_map_colored.ply", pc)

                if len(imu_trajectory) > 0:
                    imu_pc = o3d.geometry.PointCloud()
                    imu_pc.points = o3d.utility.Vector3dVector(np.array(imu_trajectory))
                    imu_pc.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 0.0], (len(imu_trajectory), 1)))
                    o3d.io.write_point_cloud("output/imu_trajectory.ply", imu_pc)

                print("Map + trajectory saved to /output/")
                np.save("output/lidar.npy", np.array(lidar_map_points))
                np.save("output/radar.npy", np.array(radar_map_points))
                np.save("output/nn_lidar.npy", np.array(nn_lidar_points))
                np.save("output/imu_trajectory.npy", np.array(imu_trajectory))

            except Exception as e:
                print(f"Error saving map: {e}")
            time.sleep(5)

    def camera_cb(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = array[:, :, :3]
        pil_img = Image.fromarray(rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        camera_image['data'] = f"data:image/jpeg;base64,{img_str}"

    camera.listen(camera_cb)

    def lidar_cb(data):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        latest_lidar['points'] = points

    def radar_cb(radar_data):
        pts = []
        for d in radar_data:
            x = d.depth * np.cos(d.altitude) * np.cos(d.azimuth)
            y = d.depth * np.cos(d.altitude) * np.sin(d.azimuth)
            z = d.depth * np.sin(d.altitude)
            pts.append((x, y, z))
        latest_radar['points'].extend(pts)

    lidar.listen(lidar_cb)
    radar1.listen(radar_cb)
    radar2.listen(radar_cb)
    radar3.listen(radar_cb)

    print("SLAM backend running...")
    os.makedirs("output", exist_ok=True)
    threading.Thread(target=save_map_continuously, daemon=True).start()

    try:
        while True:
            transform = vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            yaw = np.radians(rotation.yaw)
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            t = np.array([location.x, location.y, location.z])
            latest_radar['points'] = []
            transform = vehicle.get_transform()
            true_location = transform.location

            noisy_location = carla.Location(
                x=true_location.x + np.random.normal(0, pos_noise_std),
                y=true_location.y + np.random.normal(0, pos_noise_std),
                z=true_location.z + np.random.normal(0, 0.02) + 0.5,  # Adjust z to account for vehicle height
            )
            imu_trajectory.append((noisy_location.x, noisy_location.y, noisy_location.z))

            time.sleep(0.05)
            lidar_points = latest_lidar['points']
            radar_points = latest_radar['points']

            if lidar_points is not None and len(lidar_points) > 150:
                # Transform LiDAR points to vehicle frame
                lidar_vehicle = lidar_points - lidar_offset
                # Transform to world coordinates
                transformed = lidar_vehicle @ R.T + t
                lidar_map_points.extend(transformed.tolist())

            if radar_points is not None and len(radar_points) > 3:
                radar_np = np.array(radar_points).astype(np.float32)
                # Assume radar points are from radar1 for simplicity (combine multiple radars if needed)
                radar_vehicle = radar_np - radar1_offset
                # Transform to world coordinates for radar_map_points
                radar_world = radar_vehicle @ R.T + t
                radar_map_points.extend(radar_world.tolist())

                # Prepare radar points for neural network (in vehicle frame)
                if len(radar_vehicle) > 256:
                    indices = np.random.choice(len(radar_vehicle), 256, replace=False)
                    radar_vehicle = radar_vehicle[indices]
                elif len(radar_vehicle) < 256:
                    radar_vehicle = np.pad(radar_vehicle, ((0, 256 - len(radar_vehicle)), (0, 0)), mode='constant')

                # Convert radar to LiDAR-like point cloud using the neural network
                radar_tensor = torch.from_numpy(radar_vehicle).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_lidar = model(radar_tensor).squeeze(0).cpu().numpy()

                # Transform predicted points to world coordinates
                transformed = pred_lidar @ R.T + t
                nn_lidar_points.extend(transformed.tolist())

            time.sleep(0.1)

    except KeyboardInterrupt:
        lidar.stop()
        radar1.stop()
        radar2.stop()
        radar3.stop()
        vehicle.destroy()

if __name__ == "__main__":
    carla_slam_loop()