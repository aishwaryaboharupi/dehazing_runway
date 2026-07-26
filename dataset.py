import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from huggingface_hub import snapshot_download

class CockpitDehazeDataset(Dataset):
    def __init__(self, repo_id="NeuroPropel/CockpitAI_dehaze_dataset", split="train", crop_size=256, token=None):
        self.crop_size = crop_size
        self.split = split
        
        hf_token = token or os.getenv("HF_TOKEN")
        
        # Download ONLY the required folder (Skips 150,000 old/junk commit files!)
        if split == "train":
            allow_patterns = "filtered_train/*"
        else:
            allow_patterns = "test_data/*"

        print(f"Downloading ONLY {split} data ({allow_patterns})...")
        self.local_dir = snapshot_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            token=hf_token,
            allow_patterns=allow_patterns
        )
        
        self.pairs = []
        if split == "train":
            base_path = os.path.join(self.local_dir, "filtered_train")
            for root, _, files in os.walk(base_path):
                if "hazy" in root:
                    clean_root = root.replace("hazy", "clean")
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            hazy_p = os.path.join(root, f)
                            clean_p = os.path.join(clean_root, f)
                            if os.path.exists(clean_p):
                                self.pairs.append((hazy_p, clean_p))
        else:
            test_hazy = os.path.join(self.local_dir, "test_data", "hazy")
            test_clean = os.path.join(self.local_dir, "test_data", "clean")
            if os.path.exists(test_hazy):
                for f in os.listdir(test_hazy):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.pairs.append((os.path.join(test_hazy, f), os.path.join(test_clean, f)))

        print(f"SUCCESS! Loaded {len(self.pairs)} clean image pairs for {split} split.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_p, clean_p = self.pairs[idx]
        hazy_img = Image.open(hazy_p).convert("RGB")
        clean_img = Image.open(clean_p).convert("RGB")

        if self.split == "train":
            i, j, h, w = T.RandomCrop.get_params(hazy_img, output_size=(self.crop_size, self.crop_size))
            hazy_img = T.functional.crop(hazy_img, i, j, h, w)
            clean_img = T.functional.crop(clean_img, i, j, h, w)

            if torch.rand(1) > 0.5:
                hazy_img = T.functional.hflip(hazy_img)
                clean_img = T.functional.hflip(clean_img)

        transform = T.Compose([T.ToTensor()])
        return transform(hazy_img), transform(clean_img)