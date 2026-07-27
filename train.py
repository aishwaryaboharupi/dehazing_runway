import os
import sys
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from datasets import load_dataset
from models import get_model

class HFDehazeIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, img_size=(256, 256)):
        self.hf_dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __iter__(self):
        for item in self.hf_dataset:
            hazy_tensor = self.transform(item['hazy'].convert("RGB"))
            clear_tensor = self.transform(item['clear'].convert("RGB"))
            yield hazy_tensor, clear_tensor

def train(model_name="mamba", epochs=30, batch_size=8, lr=1e-3, save_every=5):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using Device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)

    # Adjust batch size automatically for VRAM-heavy models
    if model_name in ["transformer", "mamba"] and batch_size > 8:
        batch_size = 8

    model = get_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"--> Loading dataset stream for target model: '{model_name}'...")
    hf_ds = load_dataset("NeuroPropel/CockpitAI_dehaze_clean", split="train", streaming=True)
    
    dataset = HFDehazeIterableDataset(hf_ds)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    print(f"--> Starting dedicated training for {epochs} epochs on [{model_name.upper()}]...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        step = 0
        
        for hazy_imgs, clear_imgs in loader:
            hazy_imgs = hazy_imgs.to(device)
            clear_imgs = clear_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(hazy_imgs)
            loss = criterion(outputs, clear_imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] | Step {step} | Loss: {running_loss/step:.4f}")

        avg_loss = running_loss / max(step, 1)
        print(f"=== Epoch [{epoch}/{epochs}] Complete | Avg Loss: {avg_loss:.4f} ===\n")

        # Save checkpoint every 'save_every' epochs and as final model file
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = f"checkpoints/{model_name}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), f"model_{model_name}.pth")
            print(f"--> Saved checkpoint: {ckpt_path}\n")

    print(f"--> Dedicated {model_name.upper()} Training Complete!")

if __name__ == "__main__":
    # Allows command-line arguments: python train.py <model_name> <epochs>
    target_model = sys.argv[1] if len(sys.argv) > 1 else "mamba"
    target_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    train(model_name=target_model, epochs=target_epochs)