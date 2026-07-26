import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from datasets import load_dataset

class CockpitDehazeDataset(Dataset):
    def __init__(self, repo_id="NeuroPropel/CockpitAI_dehaze_dataset", split="train", crop_size=256, token=None):
        self.crop_size = crop_size
        self.split = split
        hf_token = token or os.getenv("HF_TOKEN")
        
        print(f"Loading {split} split via Hugging Face Datasets API...")
        
        # Stream image metadata directly without downloading git commit trees
        if split == "train":
            self.dataset = load_dataset(repo_id, data_dir="filtered_train", token=hf_token, split="train")
        else:
            self.dataset = load_dataset(repo_id, data_dir="test_data", token=hf_token, split="train")

        print(f"SUCCESS! Loaded {len(self.dataset)} image pairs for {split} split.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load PIL images directly from HF dataset object
        hazy_img = item["hazy"].convert("RGB") if isinstance(item["hazy"], Image.Image) else Image.open(item["hazy"]).convert("RGB")
        clean_img = item["clean"].convert("RGB") if isinstance(item["clean"], Image.Image) else Image.open(item["clean"]).convert("RGB")

        # Dynamic Augmentations for Training
        if self.split == "train":
            i, j, h, w = T.RandomCrop.get_params(hazy_img, output_size=(self.crop_size, self.crop_size))
            hazy_img = T.functional.crop(hazy_img, i, j, h, w)
            clean_img = T.functional.crop(clean_img, i, j, h, w)

            if torch.rand(1) > 0.5:
                hazy_img = T.functional.hflip(hazy_img)
                clean_img = T.functional.hflip(clean_img)

        transform = T.Compose([T.ToTensor()])
        return transform(hazy_img), transform(clean_img)