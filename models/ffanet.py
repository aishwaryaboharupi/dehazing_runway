import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class PixelAttention(nn.Module):
    def __init__(self, channel):
        super(PixelAttention, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.pa(x)

class Block(nn.Module):
    def __init__(self, channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.ca = ChannelAttention(channel)
        self.pa = PixelAttention(channel)

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        res = self.ca(res)
        res = self.pa(res)
        return x + res

class Group(nn.Module):
    def __init__(self, channel, num_blocks=3):
        super(Group, self).__init__()
        modules = [Block(channel) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*modules)
        self.conv = nn.Conv2d(channel, channel, 3, padding=1)

    def forward(self, x):
        res = self.blocks(x)
        res = self.conv(res)
        return x + res

class FFANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, num_groups=3):
        super(FFANet, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.groups = nn.ModuleList([Group(dim) for _ in range(num_groups)])
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * num_groups, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        self.out_conv = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x):
        feat = self.in_conv(x)
        group_outputs = []
        curr = feat
        for group in self.groups:
            curr = group(curr)
            group_outputs.append(curr)
        fused = self.fusion(torch.cat(group_outputs, dim=1))
        out = self.out_conv(fused + feat)
        return torch.clamp(out, 0.0, 1.0)