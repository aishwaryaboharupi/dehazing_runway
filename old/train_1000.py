import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

class SurvivalDehazingDataset(Dataset):
    def __init__(self, hazy_dir, clean_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir
        
        # Slicing Hack: Takes every 60th image. 
        # Out of 59,805 images, this gives us exactly ~1,000 perfectly diverse frames instantly!
        self.hazy_images = sorted(os.listdir(hazy_dir))[::60]
        self.clean_images = sorted(os.listdir(clean_dir))[::60]
        self.transform = transform

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        
        hazy_img = Image.open(hazy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clean_img = self.transform(clean_img)
            
        return hazy_img, clean_img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Survival Pipeline on: {device}")

    HAZY_DIR = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\hazy"
    CLEAN_DIR = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\clean"

    # Lightweight 256x256 resolution for ultra-fast training
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = SurvivalDehazingDataset(hazy_dir=HAZY_DIR, clean_dir=CLEAN_DIR, transform=transform_pipeline)
    
    # num_workers=0 is 100% safe on Windows to prevent crashes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )

    try:
        from old.model import AODNet
        model = AODNet().to(device)
    except ImportError:
        print("Error: Could not find AODNet in model.py.")
        return

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    writer = SummaryWriter(log_dir='runs/survival_run')
    
    # We only need 10 to 15 epochs to show a beautiful descending curve!
    epochs = 15
    print(f"\n--- PILOT RUN INITIALIZED ---")
    print(f"Using a smart subset of {len(train_dataset)} images.")
    print(f"Targeting {epochs} epochs. Estimate time: ~10 minutes.\n")

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (hazy, clean) in progress_bar:
            hazy, clean = hazy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            output = model(hazy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            writer.add_scalar('Loss/Batch', loss.item(), global_step)
            global_step += 1
            
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/Epoch', epoch_loss, epoch)

    # Save pilot weights
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/aodnet_pilot_success.pt')
    writer.close()
    print("\n--- PILOT SUCCESS ---")
    print("Training finished! Open TensorBoard to view your beautiful curves.")

if __name__ == '__main__':
    main()