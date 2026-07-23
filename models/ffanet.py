import torch
import torch.nn as nn

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.pa(x)

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.ca(self.avg_pool(x))

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, padding=kernel_size // 2, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, padding=kernel_size // 2, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        return res + x

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size, padding=kernel_size // 2))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.gp(x)

class FFANet(nn.Module):
    def __init__(self, gps=3, blocks=4, conv=nn.Conv2d):
        super(FFANet, self).__init__()
        self.gps = gps
        self.dim = 64
        
        self.pre = conv(3, self.dim, 3, padding=1)
        self.g1 = Group(conv, self.dim, 3, blocks)
        self.g2 = Group(conv, self.dim, 3, blocks)
        self.g3 = Group(conv, self.dim, 3, blocks)
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * gps, self.dim // 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 4, self.dim * gps, 1, padding=0),
            nn.Sigmoid()
        )
        self.palayer = PALayer(self.dim)
        self.post = nn.Sequential(
            conv(self.dim, self.dim, 3, padding=1),
            conv(self.dim, 3, 3, padding=1)
        )

    def forward(self, x):
        res = self.pre(x)
        res1 = self.g1(res)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim, 1, 1)
        
        out = w[:, 0] * res1 + w[:, 1] * res2 + w[:, 2] * res3
        out = self.palayer(out)
        out = self.post(out + res)
        return out