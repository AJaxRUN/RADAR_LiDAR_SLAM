import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from pointnet_completion import PointNetCompletion, chamfer_distance

class RadarLidarDataset(Dataset):
    def __init__(self, dataset_dir="dataset", max_samples=1000):
        self.radar_files = [f"dataset/radar_{i}.npy" for i in range(max_samples) if os.path.exists(f"dataset/radar_{i}.npy")]
        self.lidar_files = [f"dataset/lidar_{i}.npy" for i in range(max_samples) if os.path.exists(f"dataset/lidar_{i}.npy")]
        assert len(self.radar_files) == len(self.lidar_files), "Mismatched radar and LiDAR files"

    def __len__(self):
        return len(self.radar_files)

    def __getitem__(self, idx):
        radar = np.load(self.radar_files[idx]).astype(np.float32)
        lidar = np.load(self.lidar_files[idx]).astype(np.float32)
        return radar, lidar

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetCompletion(input_points=256, output_points=2048).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = RadarLidarDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    num_epochs = 200
    losses = []

    plt.ion() 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Training Loss Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Chamfer Distance Loss')
    ax.grid(True)
    line, = ax.plot([], [], 'b-', label='Training Loss')
    ax.legend()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for radar, lidar in dataloader:
            radar, lidar = radar.to(device), lidar.to(device)
            noise = torch.normal(mean=0.0, std=0.05, size=radar.shape, device=device)
            radar = radar + noise
            optimizer.zero_grad()
            pred = model(radar)
            loss = chamfer_distance(pred, lidar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, (Loss: {avg_loss:.4f})")

        line.set_xdata(range(1, epoch + 2))
        line.set_ydata(losses)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    torch.save(model.state_dict(), "pointnet_completion.pth")

    plt.savefig('training_loss_plot.png')
    plt.ioff() 
    plt.close()

if __name__ == "__main__":
    train_model()