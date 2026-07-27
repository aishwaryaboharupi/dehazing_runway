import torch
import torch.nn as nn

class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        
        cat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(cat1))
        
        cat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(cat2))
        
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = self.relu(self.conv5(cat3))
        
        if k.size() != x.size():
            raise Exception("K map and image size mismatch")
            
        output = k * x - k + 1.0
        return self.relu(output)