import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import sys

# --- 1. FFA-NET ATTENTION SUB-MODULES ---

class PixelAttention(nn.Module):
    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.conv(x)
        return x * weights

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.conv(self.gap(x))
        return x * weights

class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels)
        self.pa = PixelAttention(channels)
        
    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        res = self.ca(res)
        res = self.pa(res)
        return x + res

# --- 2. CORE FFA-NET ARCHITECTURE ---

class FFANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, blocks=3):
        super(FFANet, self).__init__()
        self.g1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.group = nn.ModuleList([ResidualAttentionBlock(64) for _ in range(blocks)])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * blocks, 64, 1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(64)
        self.pa = PixelAttention(64)
        
        self.g2 = nn.Conv2d(64, 64, 3, padding=1)
        self.g3 = nn.Conv2d(64, out_channels, 3, padding=1)
        
    def forward(self, x):
        feat = self.g1(x)
        group_outputs = []
        out = feat
        for block in self.group:
            out = block(out)
            group_outputs.append(out)
            
        fused = self.fusion(torch.cat(group_outputs, dim=1))
        fused = self.pa(self.ca(fused))
        
        out = self.g3(self.g2(fused) + feat)
        return torch.clamp(out + x, 0.0, 1.0)

# --- 3. IN-MEMORY RUNWAY FOG SIMULATOR WITH DOWN-SCALING ---

class FFARunwayDataset(Dataset):
    def __init__(self, clean_dir):
        self.clean_dir = clean_dir
        self.images = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.clean_dir, img_name)
        
        # Load and dynamically downscale to a memory-safe footprint
        raw_img = cv2.imread(img_path)
        resized_img = cv2.resize(raw_img, (256, 256))
        clean_img = resized_img / 255.0
        
        clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).float()
        
        beta = 0.12
        tx = np.exp(-beta * 15.0)
        foggy_img = clean_img * tx + 0.8 * (1.0 - tx)
        foggy_tensor = torch.from_numpy(foggy_img).permute(2, 0, 1).float()
        
        return foggy_tensor, clean_tensor

# --- 4. ENGINE TRAINING ORCHESTRATOR ---

def run_ffa_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"[INFO] Initializing FFA-Net Core Framework on: {device}")
    print(f"==================================================")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if not os.path.exists("clean_images") or len(os.listdir("clean_images")) == 0:
        print("[ERROR] 'clean_images' folder not found or empty.")
        return

    dataset = FFARunwayDataset(clean_dir="clean_images")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = FFANet(blocks=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    os.makedirs("weights", exist_ok=True)
    
    epochs = 50  
    total_imgs = len(dataset)
    print(f"[INFO] Loading {total_imgs} runway sets mapped to 256x256 resolution...")
    print(f"[INFO] Beginning FFA Attention training process...\n")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (foggy, clean) in enumerate(dataloader):
            foggy, clean = foggy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            outputs = model(foggy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * foggy.size(0)
            
            # Real-time counter line so you know it is moving!
            sys.stdout.write(f"\rEpoch [{epoch+1:02d}/{epochs}] | Processing Image: {i+1}/{total_imgs}...")
            sys.stdout.flush()
            
        epoch_loss = running_loss / total_imgs
        print(f"\n ➔ Milestone Completed: Epoch [{epoch+1:02d}/{epochs}] | MSE Loss: {epoch_loss:.6f}\n")
        
    torch.save(model.state_dict(), "weights/ffanet_scratch.pth")
    print(f"\n[SUCCESS] High-fidelity parameters stored: 'weights/ffanet_scratch.pth'\n")

if __name__ == "__main__":
    run_ffa_training()