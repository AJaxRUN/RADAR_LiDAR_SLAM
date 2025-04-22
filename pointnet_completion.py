import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetCompletion(nn.Module):
    def __init__(self, input_points=256, output_points=2048):
        super(PointNetCompletion, self).__init__()
        self.input_points = input_points
        self.output_points = output_points

        # Encoder: PointNet feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Global feature
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(1024)

        # Decoder: Generate dense point cloud
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, output_points * 3)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.bn_fc4 = nn.BatchNorm1d(256)

    def forward(self, x):
        # Input: (batch_size, input_points, 3)
        x = x.transpose(1, 2)  # (batch_size, 3, input_points)

        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = torch.max(x, 2)[0]  # (batch_size, 256)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))

        # Decoder
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = F.relu(self.bn_fc4(self.fc4(x)))
        x = self.fc5(x)  # (batch_size, output_points * 3)

        # Reshape to point cloud
        x = x.view(-1, self.output_points, 3)  # (batch_size, output_points, 3)
        return x

def chamfer_distance(pred, gt):
    # pred, gt: (batch_size, num_points, 3)
    pred = pred.unsqueeze(2)  # (batch_size, num_points, 1, 3)
    gt = gt.unsqueeze(1)  # (batch_size, 1, num_points, 3)
    dist = torch.sum((pred - gt) ** 2, dim=-1)  # (batch_size, num_points, num_points)
    loss1 = torch.mean(torch.min(dist, dim=1)[0])
    loss2 = torch.mean(torch.min(dist, dim=2)[0])
    return loss1 + loss2