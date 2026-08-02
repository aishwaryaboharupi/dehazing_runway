import os
import sys
import gc
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from models import get_model


class HFDehazeDataset(Dataset):
    def __init__(self, hf_dataset, img_size=(256, 256)):
        self.hf_dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        hazy_tensor = self.transform(item['hazy'].convert("RGB"))
        clear_tensor = self.transform(item['clear'].convert("RGB"))
        return hazy_tensor, clear_tensor


def find_latest_checkpoint(model_name):
    candidates = glob.glob(f"checkpoints/{model_name}_resume_epoch_*.pth")
    if not candidates:
        return None
    candidates.sort(key=lambda p: int(p.split("_epoch_")[-1].split(".pth")[0]))
    return candidates[-1]


def train(model_name="mamba", epochs=30, batch_size=8, lr=1e-3, save_every=1):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    if model_name in ["transformer", "mamba"] and batch_size > 8:
        batch_size = 8

    model = get_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    start_epoch = 1

    ckpt_path = find_latest_checkpoint(model_name)
    if ckpt_path is not None:
        print(f"--> Found existing checkpoint: {ckpt_path}. Resuming...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"--> Resuming from epoch {start_epoch}/{epochs}")
    else:
        print("--> No checkpoint found. Starting fresh.")

    if start_epoch > epochs:
        print(f"--> Training already complete for {model_name} ({epochs} epochs). Nothing to do.")
        return

    print(f"--> Loading dataset for target model: '{model_name}'...")
    hf_ds = load_dataset("NeuroPropel/CockpitAI_dehaze_clean", split="train", streaming=False)
    dataset = HFDehazeDataset(hf_ds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=True)

    print(f"--> Starting training for {model_name.upper()} (epoch {start_epoch} -> {epochs})...\n")

    for epoch in range(start_epoch, epochs + 1):
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

        if epoch % save_every == 0 or epoch == epochs:
            resume_path = f"checkpoints/{model_name}_resume_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
            }, resume_path)
            torch.save(model.state_dict(), f"model_{model_name}.pth")
            print(f"--> Saved resumable checkpoint: {resume_path}\n")

    print(f"--> Training Complete for {model_name.upper()}!")


if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else "mamba"
    target_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    train(model_name=target_model, epochs=target_epochs)