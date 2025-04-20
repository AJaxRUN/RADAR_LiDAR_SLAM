# slam_dashboard.py
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
from slam_backend import trajectory
import threading
from slam_backend import carla_slam_loop
from slam_backend import camera_image
from slam_backend import lidar_map_points, radar_map_points

# Launch SLAM thread
threading.Thread(target=carla_slam_loop, daemon=True).start()

app = Dash(__name__)
app.title = "Live SLAM Viewer"

app.layout = html.Div([
    html.H3("CARLA SLAM: RGB View + Trajectory"),
    html.Div([
        html.Img(id="camera-view", style={"width": "48%", "border": "1px solid gray"}),
        dcc.Graph(id="live-plot", style={"width": "50%", "height": "90vh"}),
    ], style={"display": "flex", "justifyContent": "space-between"}),
    dcc.Interval(id="interval", interval=1000, n_intervals=0)
])


@app.callback(Output("live-plot", "figure"), [Input("interval", "n_intervals")])
def update_graph(n):
    from slam_backend import imu_trajectory  # Re-import inside callback
    lidar = np.array(lidar_map_points) if lidar_map_points else np.zeros((0, 3))
    radar = np.array(radar_map_points) if radar_map_points else np.zeros((0, 3))
    traj = np.array(trajectory) if trajectory else np.zeros((0, 3))
    imu_traj = np.array(imu_trajectory) if imu_trajectory else np.zeros((0, 3))
    imu_traj_trace = go.Scatter3d(
        x=imu_traj[:, 0], y=imu_traj[:, 1], z=imu_traj[:, 2],
        mode="lines+markers", marker=dict(size=2, color="black"),
        name="IMU Trajectory"
    )
    lidar_trace = go.Scatter3d(
        x=lidar[:, 0], y=lidar[:, 1], z=lidar[:, 2],
        mode="markers", marker=dict(size=1, color="blue"),
        name="LiDAR"
    )
    radar_trace = go.Scatter3d(
        x=radar[:, 0], y=radar[:, 1], z=radar[:, 2],
        mode="markers", marker=dict(size=1, color="green"),
        name="RADAR"
    )
    traj_trace = go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
        mode="lines+markers", marker=dict(size=2, color="red"),
        name="Trajectory"
    )
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    )
    return go.Figure(data=[lidar_trace, radar_trace, traj_trace, imu_traj_trace], layout=layout)

@app.callback(
    Output("camera-view", "src"),
    [Input("interval", "n_intervals")]
)
def update_image(n):
    return camera_image.get("data", "")


if __name__ == "__main__":
    app.run_server(debug=False)
