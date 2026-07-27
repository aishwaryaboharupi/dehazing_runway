import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    """Multi-Dhead Transposed Self-Attention"""
    def __init__(self, dim, num_heads=8):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)

class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super(GDFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # Apply Norm along channel dimension
        nx = x.permute(0, 2, 3, 1).contiguous()
        nx = self.norm1(nx).permute(0, 3, 1, 2).contiguous()
        x = x + self.attn(nx)

        nx = x.permute(0, 2, 3, 1).contiguous()
        nx = self.norm2(nx).permute(0, 3, 1, 2).contiguous()
        x = x + self.ffn(nx)
        return x

class DehazeTransformer(nn.Module):
    def __init__(self, dim=48, num_blocks=4):
        super(DehazeTransformer, self).__init__()
        self.in_proj = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.in_proj(x)
        feat = self.blocks(feat)
        out = self.out_proj(feat)
        return torch.sigmoid(out)