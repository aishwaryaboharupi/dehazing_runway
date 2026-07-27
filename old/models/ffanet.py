import os, sys, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from old.dataset import FastDehazeDataset
from huggingface_hub import login

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Hugging Face Login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== [MODEL 2/2] Training FFA-Net on {device} ===")

# 2. Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
NUM_SAMPLES = 2500

# 3. Dataset & Dataloader
train_dataset = FastDehazeDataset(token=HF_TOKEN, num_samples=NUM_SAMPLES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# 4. FFA-Net Architecture Definition
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

# 5. Initialize Model, Loss, Optimizer
model = FFANet().to(device)

save_dir = "/content/drive/MyDrive/CockpitAI_Weights/FFANet" if os.path.exists("/content/drive/MyDrive") else "checkpoints/FFANet"
os.makedirs(save_dir, exist_ok=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Training Loop
print(f"Starting FFA-Net training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (hazy, clean) in enumerate(train_loader):
        hazy, clean = hazy.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = model(hazy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if (i + 1) % 50 == 0:
            print(f"[FFA-Net] Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 50:.5f}")
            running_loss = 0.0

    checkpoint_path = os.path.join(save_dir, f"ffanet_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

print("FFA-Net Training Complete!")