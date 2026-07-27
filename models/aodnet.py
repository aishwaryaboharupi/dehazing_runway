import torch
import torch.nn as nn

class AODNet(nn.Module):
    """
    All-in-One Dehazing Network (AOD-Net)
    Reformulates the atmospheric scattering model: K(x) = b * I(x) - b + 1
    """
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, 5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, 7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        c3 = self.relu(self.conv3(torch.cat([c1, c2], dim=1)))
        c4 = self.relu(self.conv4(torch.cat([c2, c3], dim=1)))
        k = self.relu(self.conv5(torch.cat([c1, c2, c3, c4], dim=1)))
        
        output = k * x - k + 1.0
        return torch.clamp(output, 0.0, 1.0)