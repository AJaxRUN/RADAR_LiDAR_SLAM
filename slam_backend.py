# slam_backend.py
import carla
import numpy as np
import threading
import time
import base64
from io import BytesIO
from PIL import Image
import os
import open3d as o3d


imu_trajectory = []
camera_image = {'data': None}
# Standard deviation of simulated IMU noise
pos_noise_std = 0.2

lidar_map_points = []
radar_map_points = []
trajectory = []

def carla_slam_loop():
    global trajectory

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    # Load Town02
    if world.get_map().name != "Town02":
        print("Loading Town02...")
        client.load_world("Town02")
        time.sleep(5)
        world = client.get_world()  # Reload after switching
        bp_lib = world.get_blueprint_library()
        
        #Weather - Fog
        weather = carla.WeatherParameters(
            fog_density=75,          # Max fog density (0.0 to 1.0)
            fog_distance=2.0,         # Minimum visibility distance (2m means thick fog)
            cloudiness=100.0,         # Fully overcast (for a dull scene)
            precipitation=0.0,        # No rain
            sun_altitude_angle=10.0   # Low sunlight = dull lighting
        )
        world.set_weather(weather)
        print("Extreme fog applied.")
        fog = world.get_weather().fog_density
        print(f"Fog density in world: {fog}")

    # Make all traffic lights green
    for light in world.get_actors().filter("traffic.traffic_light"):
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(9999)
        light.freeze(True)

    print("🚦 All traffic lights set to GREEN.")


    # Destroy other actors
    for actor in world.get_actors().filter("*vehicle*"):
        actor.destroy()
    for actor in world.get_actors().filter("*walker*"):
        actor.destroy()
    print("✅ Cleared all other actors.")

    # Vehicle
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn)
    vehicle.set_autopilot(False)

    tf = carla.Transform(carla.Location(x=0, z=2.5))

    # LiDAR
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "100.0")
    lidar_bp.set_attribute("channels", "128")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("points_per_second", "240000") 
    lidar = world.spawn_actor(lidar_bp, tf, attach_to=vehicle)

    # RADAR
    radar_bp = bp_lib.find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", "60")
    radar_bp.set_attribute("vertical_fov", "20")
    radar_bp.set_attribute("range", "100")

    # RADAR 1 – Front-center (main)
    radar1_tf = carla.Transform(carla.Location(x=1.0, y=0.0, z=2.0))
    radar1 = world.spawn_actor(radar_bp, radar1_tf, attach_to=vehicle)

    # RADAR 2 – Extended forward
    radar2_tf = carla.Transform(carla.Location(x=1.0, y=0.5, z=2.0))  # Same Y
    radar2 = world.spawn_actor(radar_bp, radar2_tf, attach_to=vehicle)

    # RADAR 3 – Closer to center/back
    radar3_tf = carla.Transform(carla.Location(x=1.0, y=-0.5, z=2.0))  # Same Y
    radar3 = world.spawn_actor(radar_bp, radar3_tf, attach_to=vehicle)

    spectator = world.get_spectator()
    def follow_vehicle():
        prev_location = None

        while True:
            transform = vehicle.get_transform()
            forward = transform.get_forward_vector()
            right = transform.get_right_vector()
            up = transform.get_up_vector()

            # Offset camera from vehicle
            dx, dy, dz = -6, 0, 3
            target_loc = transform.location + forward * dx + right * dy + up * dz

            # Smooth interpolation
            if prev_location is None:
                smoothed_loc = target_loc
            else:
                alpha = 0.1  # smoothing factor (0 = frozen, 1 = instant jump)
                smoothed_loc = prev_location * (1 - alpha) + target_loc * alpha

            prev_location = smoothed_loc

            # Rotation: match yaw, slight downward pitch
            cam_rotation = carla.Rotation(
                pitch=-15.0,
                yaw=transform.rotation.yaw,
                roll=0.0
            )

            spectator.set_transform(carla.Transform(smoothed_loc, cam_rotation))
            time.sleep(0.05)

    threading.Thread(target=follow_vehicle, daemon=True).start()

    vehicle.set_autopilot(True)


    tm = client.get_trafficmanager()
    tm.ignore_lights_percentage(vehicle, 100.0)
    tm.ignore_signs_percentage(vehicle, 100.0)
    tm.ignore_vehicles_percentage(vehicle, 100.0)
    tm.set_global_distance_to_leading_vehicle(5.0)
    tm.set_synchronous_mode(False)

    latest_lidar = {'points': None}
    latest_radar = {'points': []}

    # Camera Sensor
    camera_bp = bp_lib.find("sensor.camera.rgb")

    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.05")
    camera_bp.set_attribute("enable_postprocess_effects", "True")


    camera_tf = carla.Transform(carla.Location(x=1.5, z=3.0))  # Higher Z

    camera = world.spawn_actor(camera_bp, camera_tf, attach_to=vehicle)

    def save_map_continuously():
        while True:
            try:
                # Save separate clouds with color coding
                if lidar_map_points or radar_map_points:
                    combined = []
                    colors = []

                    if lidar_map_points:
                        lidar_np = np.array(lidar_map_points)
                        combined.append(lidar_np)
                        colors.append(np.tile([0, 0.6, 1.0], (len(lidar_np), 1)))  # blue

                    if radar_map_points:
                        radar_np = np.array(radar_map_points)
                        combined.append(radar_np)
                        colors.append(np.tile([0.2, 1.0, 0.0], (len(radar_np), 1)))  # green

                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(np.vstack(combined))
                    pc.colors = o3d.utility.Vector3dVector(np.vstack(colors))
                    o3d.io.write_point_cloud("output/slam_map_colored.ply", pc)

                # Save IMU trajectory in black
                if len(imu_trajectory) > 0:
                    imu_pc = o3d.geometry.PointCloud()
                    imu_pc.points = o3d.utility.Vector3dVector(np.array(imu_trajectory))
                    imu_pc.colors = o3d.utility.Vector3dVector(
                        np.tile([0.0, 0.0, 0.0], (len(imu_trajectory), 1)))  # Black
                    o3d.io.write_point_cloud("output/imu_trajectory.ply", imu_pc)

                # Save trajectory
                # if len(trajectory) > 0:
                #     traj_pc = o3d.geometry.PointCloud()
                #     traj_pc.points = o3d.utility.Vector3dVector(np.array(trajectory))
                #     o3d.io.write_point_cloud("output/slam_trajectory.ply", traj_pc)

                print("Map + trajectory saved to /output/")
                np.save("output/lidar.npy", np.array(lidar_map_points))
                np.save("output/radar.npy", np.array(radar_map_points))
                np.save("output/trajectory.npy", np.array(trajectory))
                np.save("output/imu_trajectory.npy", np.array(imu_trajectory))

            except Exception as e:
                print(f"Error saving map: {e}")

            time.sleep(5)  # save every 5 seconds


    def camera_cb(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = array[:, :, :3]

        # Convert to base64
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

    print("🛰 SLAM backend running...")
    os.makedirs("output", exist_ok=True)
    threading.Thread(target=save_map_continuously, daemon=True).start()

    try:
        while True:
            transform = vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            # trajectory.append((location.x, location.y, location.z))
            yaw = np.radians(rotation.yaw)
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            t = np.array([location.x, location.y, location.z])
            latest_radar['points'] = []
            transform = vehicle.get_transform()
            true_location = transform.location

            # Add Gaussian noise to simulate IMU drift
            noisy_location = carla.Location(
                x=true_location.x + np.random.normal(0, pos_noise_std),
                y=true_location.y + np.random.normal(0, pos_noise_std),
                z=true_location.z + np.random.normal(0, 0.02),
            )

            # Append to simulated IMU trajectory
            imu_trajectory.append((noisy_location.x, noisy_location.y, noisy_location.z))

            time.sleep(0.05)
            points = latest_lidar['points']
            if points is not None and len(points) > 150:
                transformed = points @ R.T + t
                lidar_map_points.extend(transformed.tolist())

            elif latest_radar['points'] is not None and len(latest_radar['points']) > 3:
                transformed = np.array(latest_radar['points']) @ R.T + t
                radar_map_points.extend(transformed.tolist())

            else:
                time.sleep(0.05)
                continue

            time.sleep(0.1)

    except KeyboardInterrupt:
        lidar.stop()
        radar1.listen(radar_cb)
        radar2.listen(radar_cb)
        radar3.listen(radar_cb)
        vehicle.destroy()
