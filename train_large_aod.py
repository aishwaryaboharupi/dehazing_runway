import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm  # Visual progress tracker

# 1. High-Performance Safe Data Loader
class CockpitDehazingDataset(Dataset):
    def __init__(self, hazy_dir, clean_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir
        self.hazy_images = sorted(os.listdir(hazy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        self.transform = transform

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        
        # Lazy loading directly from disk (keeps VS Code fast)
        hazy_img = Image.open(hazy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clean_img = self.transform(clean_img)
            
        return hazy_img, clean_img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing pipeline on: {device}")

    # Your exact system dataset paths
    HAZY_DIR = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\hazy"
    CLEAN_DIR = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\clean"

    # Optimization step: Resize 1920x1080 down to 256x256 in RAM so your GPU never runs out of memory
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = CockpitDehazingDataset(hazy_dir=HAZY_DIR, clean_dir=CLEAN_DIR, transform=transform_pipeline)
    
    # Batch size bumped to 32 for maximum GPU utilization at 256x256 resolution
    # num_workers set to 0 to prevent Windows process deadlocks
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )

    # Import your AOD-Net model structure from your model file
    try:
        from model import AODNet
        model = AODNet().to(device)
    except ImportError:
        print("Error: Could not find AODNet in model.py. Please verify model.py is in this directory.")
        return

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # TensorBoard initialization
    writer = SummaryWriter(log_dir='runs/aodnet_large_scale')
    
    epochs = 50
    print(f"\n--- SUCCESS ---")
    print(f"Dataset securely hooked: {len(train_dataset)} Category III/IV frames found.")
    print("Launching large-scale background execution loop...\n")

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Wrapped with tqdm for a visual live progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (hazy, clean) in progress_bar:
            hazy, clean = hazy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            output = model(hazy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Displays the real-time loss value on the progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Streams batch data instantly to your browser dashboard
            writer.add_scalar('Loss/Batch', loss.item(), global_step)
            global_step += 1
            
        epoch_loss = running_loss / len(train_loader)
        print(f"\n--> Epoch [{epoch+1}/{epochs}] Done - Average Loss: {epoch_loss:.6f}")
        
        writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
        
        # Save structural weights securely every 10 epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), f'weights/aodnet_large_epoch_{epoch+1}.pt')

    # Save final model state
    torch.save(model.state_dict(), 'weights/aodnet_large_final.pt')
    writer.close()
    print("Training loop complete. Model weights exported cleanly.")

if __name__ == '__main__':
    main()