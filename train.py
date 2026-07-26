import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CockpitDehazeDataset
from models import AODNet
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Hyperparameters
    batch_size = 16
    epochs = 20
    lr = 1e-4

    train_dataset = CockpitDehazeDataset(split="train", crop_size=256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = AODNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (hazy, clean) in enumerate(train_loader, 1):
            hazy, clean = hazy.to(device), clean.to(device)

            optimizer.zero_grad()
            outputs = model(hazy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.5f}")

        avg_loss = running_loss / len(train_loader)
        print(f"--> Epoch {epoch} Average Loss: {avg_loss:.5f}\n")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/model1_aodnet_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()