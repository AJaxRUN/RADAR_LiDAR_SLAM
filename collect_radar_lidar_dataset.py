import carla
import numpy as np
import threading
import time
import os

# Global variables for data collection
lidar_map_points = []
radar_map_points = []
paired_data = []

def collect_radar_lidar_data():
    global lidar_map_points, radar_map_points, paired_data

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Load Town02 if not already loaded
    if world.get_map().name != "Town02":
        print("Loading Town02...")
        client.load_world("Town02")
        time.sleep(5)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # Apply fog weather
        weather = carla.WeatherParameters(
            fog_density=75, cloudiness=100.0, precipitation=0.0, sun_altitude_angle=10.0
        )
        world.set_weather(weather)
        print("Fog weather applied.")

    # Set all traffic lights to green
    for light in world.get_actors().filter("traffic.traffic_light"):
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(9999)
        light.freeze(True)
    print("All traffic lights set to GREEN.")

    # Clear other actors
    for actor in world.get_actors().filter("*vehicle*"):
        actor.destroy()
    for actor in world.get_actors().filter("*walker*"):
        actor.destroy()
    print("Cleared all other actors.")

    # Spawn vehicle
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn)
    vehicle.set_autopilot(True)

    # Setup sensors
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
    radar_tf = carla.Transform(carla.Location(x=1.0, y=0.0, z=2.0))
    radar = world.spawn_actor(radar_bp, radar_tf, attach_to=vehicle)

    latest_lidar = {'points': None}
    latest_radar = {'points': []}

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
        latest_radar['points'] = pts

    lidar.listen(lidar_cb)
    radar.listen(radar_cb)

    # Sensor offsets relative to vehicle center
    lidar_offset = np.array([0.0, 0.0, 2.5])  # LiDAR at (x=0, y=0, z=2.5)
    radar_offset = np.array([1.0, 0.0, 2.0])  # Radar at (x=1.0, y=0.0, z=2.0)

    # Data collection loop
    os.makedirs("dataset", exist_ok=True)
    sample_count = 0
    max_samples = 1000  # Collect 1000 paired samples

    print("Starting data collection...")
    try:
        while sample_count < max_samples:
            transform = vehicle.get_transform()
            yaw = np.radians(transform.rotation.yaw)
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            t = np.array([transform.location.x, transform.location.y, transform.location.z])

            lidar_points = latest_lidar['points']
            radar_points = latest_radar['points']

            if lidar_points is not None and len(lidar_points) > 150 and len(radar_points) > 3:
                # Transform points to vehicle coordinate frame
                lidar_vehicle = lidar_points - lidar_offset
                radar_vehicle = np.array(radar_points) - radar_offset

                # Transform to world coordinates
                lidar_transformed = lidar_vehicle @ R.T + t
                radar_transformed = radar_vehicle @ R.T + t

                # Subsample LiDAR to 2048 points for consistency
                if len(lidar_transformed) > 2048:
                    indices = np.random.choice(len(lidar_transformed), 2048, replace=False)
                    lidar_transformed = lidar_transformed[indices]
                elif len(lidar_transformed) < 2048:
                    lidar_transformed = np.pad(lidar_transformed, ((0, 2048 - len(lidar_transformed)), (0, 0)), mode='constant')

                # Subsample radar to 256 points
                if len(radar_transformed) > 256:
                    indices = np.random.choice(len(radar_transformed), 256, replace=False)
                    radar_transformed = radar_transformed[indices]
                elif len(radar_transformed) < 256:
                    radar_transformed = np.pad(radar_transformed, ((0, 256 - len(radar_transformed)), (0, 0)), mode='constant')

                # Save paired data
                np.save(f"dataset/radar_{sample_count}.npy", radar_transformed)
                np.save(f"dataset/lidar_{sample_count}.npy", lidar_transformed)
                sample_count += 1
                print(f"Collected sample {sample_count}/{max_samples}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping data collection...")
    finally:
        lidar.stop()
        radar.stop()
        vehicle.destroy()

if __name__ == "__main__":
    collect_radar_lidar_data()