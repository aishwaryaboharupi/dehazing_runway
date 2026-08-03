import os
from datasets import load_dataset

def main():
    os.makedirs("test_samples/hazy", exist_ok=True)
    os.makedirs("test_samples/clear", exist_ok=True)
    
    print("--> Downloading 10 test samples locally once...")
    dataset = load_dataset("NeuroPropel/CockpitAI_dehaze_clean", split="train", streaming=True)
    
    for idx, sample in enumerate(dataset):
        if idx >= 10:
            break
        
        hazy_img = sample['hazy'].convert("RGB")
        clear_img = sample['clear'].convert("RGB")
        
        hazy_img.save(f"test_samples/hazy/sample_{idx}.png")
        clear_img.save(f"test_samples/clear/sample_{idx}.png")
        print(f"Saved local sample {idx}")

    print("--> Done! 10 test images saved locally in 'test_samples/'")

if __name__ == "__main__":
    main()