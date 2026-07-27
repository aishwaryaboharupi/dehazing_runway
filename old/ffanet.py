import torch
import torch.nn as nn

# --- Channel Attention (CA) Layer ---
class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

# --- Pixel Attention (PA) Layer ---
class PixelAttention(nn.Module):
    def __init__(self, num_features):
        super(PixelAttention, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(num_features, num_features // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

# --- Feature Attention (FA) Block ---
class FABlock(nn.Module):
    def __init__(self, num_features):
        super(FABlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)
        self.ca = ChannelAttention(num_features)
        self.pa = PixelAttention(num_features)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        out = self.pa(out)
        return out + residual

# --- Group of FA Blocks ---
class Group(nn.Module):
    def __init__(self, num_features, num_blocks=3):
        super(Group, self).__init__()
        modules = [FABlock(num_features) for _ in range(num_blocks)]
        self.gp = nn.Sequential(*modules)
        self.conv = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.gp(x)
        out = self.conv(out)
        return out + residual

# --- Simplified FFA-Net Architecture ---
class FFANet(nn.Module):
    def __init__(self, gps=3, blocks=3, num_features=64):
        super(FFANet, self).__init__()
        self.gps = gps
        self.g1 = Group(num_features, blocks)
        self.g2 = Group(num_features, blocks)
        self.g3 = Group(num_features, blocks)
        
        # Pre-processing block
        self.pre = nn.Sequential(
            nn.Conv2d(3, num_features, 3, padding=1, bias=True)
        )
        
        # Post-processing block
        self.post = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 3, 3, padding=1, bias=True)
        )

    def forward(self, x):
        # Pre-process image to feature space
        x_feat = self.pre(x)
        
        # Group forward passes with internal residual shortcuts
        res1 = self.g1(x_feat)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        
        # Global fusion step
        fused = res1 + res2 + res3
        
        # Post-process back to RGB space
        out = self.post(fused)
        
        # Residual skip connection from the input image (Koschmieder's formulation fallback)
        return out + x