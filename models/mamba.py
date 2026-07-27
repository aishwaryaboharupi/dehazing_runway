import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualSSM2D(nn.Module):
    """
    State Space Model Block with 2D Directional Scanning
    Sweeps feature channels across horizontal & vertical axes
    """
    def __init__(self, dim):
        super(VisualSSM2D, self).__init__()
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1)
        
        # Directional Depthwise Sweeps (Left-to-Right, Top-to-Bottom)
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim)
        self.dw_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim)
        
        # State transitions
        self.state_decay = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.9)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x_proj = self.in_proj(x)
        u, gate = x_proj.chunk(2, dim=1)
        
        # Perform horizontal and vertical state-space sweeps
        scan_h = self.dw_h(u)
        scan_v = self.dw_v(u)
        
        ssm_out = (scan_h + scan_v) * self.state_decay
        activated = ssm_out * F.silu(gate)
        return self.out_proj(activated)

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super(MambaBlock, self).__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.ssm = VisualSSM2D(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )

    def forward(self, x):
        res = x + self.ssm(self.norm(x))
        out = res + self.mlp(res)
        return out

class MambaDehazeNet(nn.Module):
    def __init__(self, dim=64, depth=4):
        super(MambaDehazeNet, self).__init__()
        self.in_conv = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.layers = nn.Sequential(*[MambaBlock(dim) for _ in range(depth)])
        self.out_conv = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.in_conv(x)
        feat = self.layers(feat)
        out = self.out_conv(feat)
        return torch.sigmoid(out)