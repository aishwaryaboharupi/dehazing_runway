import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

class FastDehazeDataset(Dataset):
    def __init__(self, repo_id="NeuroPropel/CockpitAI_dehaze_dataset", token=None, num_samples=10000, transform=None):
        self.repo_id = repo_id
        self.token = token
        self.num_samples = num_samples
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        filename = f"{idx:06d}.png"
        
        try:
            hazy_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename=f"hazy/{filename}", 
                repo_type="dataset", 
                token=self.token
            )
            clean_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename=f"clean/{filename}", 
                repo_type="dataset", 
                token=self.token
            )
            
            hazy_img = Image.open(hazy_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")
        except Exception:
            hazy_img = Image.new("RGB", (256, 256))
            clean_img = Image.new("RGB", (256, 256))

        if self.transform:
            hazy_img = self.transform(hazy_img)
            clean_img = self.transform(clean_img)

        return hazy_img, clean_img