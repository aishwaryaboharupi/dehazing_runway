import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from models import get_model

# ----------------------------------------------------
# In-Memory PyTorch Dataset Wrapper (Lightning Fast)
# ----------------------------------------------------
class HFDehazeDataset(Dataset):
    def __init__(self, hf_dataset, img_size=(256, 256)):
        print("--> Pre-loading and converting images to RAM...")
        self.samples = list(hf_dataset)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        hazy_tensor = self.transform(item['hazy'].convert("RGB"))
        clear_tensor = self.transform(item['clear'].convert("RGB"))
        return hazy_tensor, clear_tensor

# ----------------------------------------------------
# Main Training Function
# ----------------------------------------------------
def train(model_name="aodnet", epochs=5, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using Device: {device}")
    
    model = get_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"--> Downloading clean dataset from Hugging Face into Colab RAM...")
    # Non-streaming download: ~520MB takes ~5-10 seconds!
    hf_ds = load_dataset("NeuroPropel/CockpitAI_dehaze_clean", split="train")
    
    dataset = HFDehazeDataset(hf_ds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print(f"--> Starting fast training for {epochs} epochs on {model_name}...\n")

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
                print(f"Epoch [{epoch}/{epochs}] | Step [{step}/{len(loader)}] | Loss: {running_loss/step:.4f}")

        avg_loss = running_loss / max(step, 1)
        print(f"=== Epoch [{epoch}/{epochs}] Complete | Avg Loss: {avg_loss:.4f} ===\n")

    save_filename = f"model_{model_name}.pth"
    torch.save(model.state_dict(), save_filename)
    print(f"--> Model weights saved to {save_filename}!")

if __name__ == "__main__":
    train(model_name="aodnet", epochs=5, batch_size=16)