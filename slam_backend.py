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


camera_image = {'data': None}

lidar_map_points = []
radar_map_points = []
trajectory = []

imu_trajectory = []
imu_velocity = np.zeros(3)
imu_position = np.zeros(3)
last_imu_time = [None] 

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

    # IMU Sensor
    imu_bp = bp_lib.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", "0.05")  # 20 Hz
    imu_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5))  # Centered
    imu_sensor = world.spawn_actor(imu_bp, imu_tf, attach_to=vehicle)

    stationary_counter = 0
    ZUPT_COUNT_THRESHOLD = 5 

    def imu_cb(imu_data):
        global imu_trajectory, imu_velocity, imu_position, last_imu_time, stationary_counter

        # --- Config ---
        ACC_SCALE = 1
        GRAVITY = np.array([0.0, 0.0, 9.8])
        ACC_NOISE_THRESH = 0.2         # m/s²
        VEL_ZUPT_THRESH = 0.05         # m/s
        MAX_VELOCITY = 50.0            # sanity clamp
        ZUPT_COUNT_THRESH = 5
        MAX_DT = 0.1

        if last_imu_time[0] is None:
            last_imu_time[0] = imu_data.timestamp
            return

        dt = imu_data.timestamp - last_imu_time[0]
        if dt <= 0 or dt > MAX_DT:
            imu_velocity[:] = 0.0
            stationary_counter = 0
            last_imu_time[0] = imu_data.timestamp
            return
        last_imu_time[0] = imu_data.timestamp

        # --- Raw Accel (scale + gravity removal) ---
        accel = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z]) * ACC_SCALE
        acc_corrected = accel - GRAVITY

        # --- Threshold low noise values ---
        acc_corrected[np.abs(acc_corrected) < ACC_NOISE_THRESH] = 0.0

        # --- Integrate acceleration to velocity ---
        imu_velocity += acc_corrected * dt

        # --- ZUPT: if acceleration is tiny for a few frames, assume stationary ---
        if np.linalg.norm(acc_corrected) < ACC_NOISE_THRESH:
            stationary_counter += 1
            if stationary_counter >= ZUPT_COUNT_THRESH:
                imu_velocity[:] = 0.0
        else:
            stationary_counter = 0

        # --- Clamp velocity (sanity) ---
        imu_velocity = np.clip(imu_velocity, -MAX_VELOCITY, MAX_VELOCITY)

        # --- Integrate velocity to position ---
        imu_position += imu_velocity * dt

        # --- Store & log ---
        imu_trajectory.append((imu_position[0], imu_position[1], imu_position[2]))

        print(f"[IMU] Accel: {acc_corrected.round(2)}, Vel: {imu_velocity.round(2)}, dt: {dt:.4f}")
        print(f"[IMU] Position: ({imu_position[1]:.2f}, {imu_position[0]:.2f}, {imu_position[2]:.2f}) m")


    imu_sensor.listen(imu_cb)
    time.sleep(1)
    vehicle.set_autopilot(True)


    latest_lidar = {'points': None}
    latest_radar = {'points': []}

    # Camera Sensor
    camera_bp = bp_lib.find("sensor.camera.rgb")

    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.1")
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


                # Save trajectory
                if len(trajectory) > 0:
                    traj_pc = o3d.geometry.PointCloud()
                    traj_pc.points = o3d.utility.Vector3dVector(np.array(trajectory))
                    o3d.io.write_point_cloud("output/slam_trajectory.ply", traj_pc)

                print("💾 Map + trajectory saved to /output/")
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
            trajectory.append((location.x, location.y, location.z))

            yaw = np.radians(rotation.yaw)
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            t = np.array([location.x, location.y, location.z])
            latest_radar['points'] = []

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
        imu_sensor.stop()
        vehicle.destroy()
