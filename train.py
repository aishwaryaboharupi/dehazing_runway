import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np

# Import the architectures we built earlier
from model import AODNet
from dataset_prep import AdverseFogSimulator

# --- 1. PYTORCH CUSTOM DATASET ---
class LardFogDataset(Dataset):
    def __init__(self, clean_dir, thickness_level='cat3a', transform=None):
        # Grab all PNG files from your clean LARD data directory
        self.clean_image_paths = glob.glob(os.path.join(clean_dir, "*.png"))
        self.thickness_level = thickness_level
        self.transform = transform

        if len(self.clean_image_paths) == 0:
            raise RuntimeError(f"No .png images found in {clean_dir}. Make sure data is loaded!")

    def __len__(self):
        return len(self.clean_image_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_image_paths[idx]
        
        # Initialize your simulation engine dynamically for this image
        simulator = AdverseFogSimulator(clean_path)
        # Generate the synthetic fog version on-the-fly
        foggy_img = simulator.generate_fog(thickness_level=self.thickness_level)
        clean_img = simulator.img  # Retain original clean BGR image

        # Convert OpenCV BGR to RGB format before passing to PyTorch
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        foggy_img = cv2.cvtColor(foggy_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            clean_img = self.transform(clean_img)
            foggy_img = self.transform(foggy_img)

        # Returns: (Input hazy tensor, Target ground-truth tensor)
        return foggy_img, clean_img

# --- 2. TRAINING ENGINE CONTEXT ---
def train_pipeline():
    # Setup paths and hyperparameters
    CLEAN_DATA_DIR = "clean_images" 
    BATCH_SIZE = 4                  # Lower this if your RTX 4050 hits Out-Of-Memory limits
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Device allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerating training via device: {device}")

    # Standard Image Transforms for Neural Networks
    # Resizing ensures batch stability; Totensor normalizes values between [0.0, 1.0]
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 640)), # Balance spatial depth and GPU processing speed
        transforms.ToTensor(),
    ])

    # Instantiate our data loaders
    dataset = LardFogDataset(clean_dir=CLEAN_DATA_DIR, thickness_level='cat3a', transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Load architecture to computing core
    model = AODNet().to(device)
    
    # Loss calculation and Optimization choices
    criterion = nn.MSELoss() # Standard pixel-to-pixel Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("Beginning AOD-Net Dehazing optimization loop...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_idx, (foggy_batch, clean_batch) in enumerate(dataloader):
            # Send mini-batches to GPU
            foggy_batch = foggy_batch.to(device)
            clean_batch = clean_batch.to(device)

            # Zero out gradients from last step
            optimizer.zero_grad()

            # Forward Pass: Feed foggy frames into AOD-Net
            dehazed_batch = model(foggy_batch)

            # Calculate error between network output and crystal clear target
            loss = criterion(dehazed_batch, clean_batch)

            # Backward Pass: Calculate gradients and adjust layer weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] ---> Mean Squared Error Loss: {epoch_loss:.5f}")

    # Save finalized trained model state dictionary
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/aodnet_cockpit.pth")
    print("Training phase complete. Model weights saved successfully at: weights/aodnet_cockpit.pth")

if __name__ == "__main__":
    train_pipeline()