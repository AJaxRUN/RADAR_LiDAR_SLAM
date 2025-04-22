from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
from carla_env import trajectory
import threading
from carla_env import carla_main_loop
from carla_env import camera_image
from carla_env import lidar_map_points, radar_map_points, nn_lidar_points

threading.Thread(target=carla_main_loop, daemon=True).start()

app = Dash(__name__)
app.title = "Live Map Viewer"

app.layout = html.Div([
    html.H3("CARLA Map: RGB View + Trajectory"),
    html.Div([
        html.Img(id="camera-view", style={"width": "48%", "border": "1px solid gray"}),
        dcc.Graph(id="live-plot", style={"width": "50%", "height": "90vh"}),
    ], style={"display": "flex", "justifyContent": "space-between"}),
    dcc.Interval(id="interval", interval=1000, n_intervals=0)
])

@app.callback(Output("live-plot", "figure"), [Input("interval", "n_intervals")])
def update_graph(n):
    from carla_env import imu_trajectory  # Re-import inside callback
    lidar = np.array(lidar_map_points) if lidar_map_points else np.zeros((0, 3))
    radar = np.array(radar_map_points) if radar_map_points else np.zeros((0, 3))
    nn_lidar = np.array(nn_lidar_points) if nn_lidar_points else np.zeros((0, 3))
    traj = np.array(trajectory) if trajectory else np.zeros((0, 3))
    imu_traj = np.array(imu_trajectory) if imu_trajectory else np.zeros((0, 3))
    imu_traj_trace = go.Scatter3d(
        x=imu_traj[:, 0], y=imu_traj[:, 1], z=imu_traj[:, 2],
        mode="lines+markers", marker=dict(size=2, color="red"),
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
    nn_lidar_trace = go.Scatter3d(
        x=nn_lidar[:, 0], y=nn_lidar[:, 1], z=nn_lidar[:, 2],
        mode="markers", marker=dict(size=1, color="purple"),
        name="NN-Generated LiDAR"
    )
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    )
    return go.Figure(data=[lidar_trace, radar_trace, imu_traj_trace, nn_lidar_trace], layout=layout)

@app.callback(
    Output("camera-view", "src"),
    [Input("interval", "n_intervals")]
)
def update_image(n):
    return camera_image.get("data", "")

if __name__ == "__main__":
    app.run_server(debug=False)