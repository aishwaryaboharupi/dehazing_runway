import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FastDehazeDataset
from huggingface_hub import login

# 1. Hugging Face Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("No HF_TOKEN environment variable provided. Proceeding with default authentication.")

# 2. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# 3. Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
NUM_SAMPLES = 5000

# 4. Data & Model Setup
train_dataset = FastDehazeDataset(token=HF_TOKEN, num_samples=NUM_SAMPLES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

try:
    from models.aod_net import AODnet
    model = AODnet().to(device)
    print("Loaded AOD-Net model structure.")
except ImportError:
    class SimpleDehazeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)
            
    model = SimpleDehazeNet().to(device)
    print("Loaded baseline model structure.")

# 5. Save path setup (Prioritize Google Drive if available)
drive_checkpoint_dir = "/content/drive/MyDrive/CockpitAI_Weights"
if os.path.exists("/content/drive/MyDrive"):
    save_dir = drive_checkpoint_dir
else:
    save_dir = "checkpoints"

os.makedirs(save_dir, exist_ok=True)
print(f"Checkpoints will be saved to: {save_dir}")

# 6. Training Loop
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
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
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 50:.5f}")
            running_loss = 0.0

    # Save checkpoint at the end of each epoch directly to Google Drive
    checkpoint_path = os.path.join(save_dir, f"aodnet_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

print("Training session finished successfully.")