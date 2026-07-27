import os
import glob
from datasets import Dataset, Image, DatasetDict

def prepare_and_upload_clean_dataset(
    root_dir="large_dataset/train", 
    hf_repo="NeuroPropel/CockpitAI_dehaze_clean"
):
    hazy_dir = os.path.join(root_dir, "hazy")
    clean_dir = os.path.join(root_dir, "clean")

    print("1. Indexing clean images...")
    clean_lookup = {}
    clean_paths = glob.glob(os.path.join(clean_dir, "**", "*.png"), recursive=True)

    for path in clean_paths:
        filename = os.path.basename(path)
        base_id = filename.replace('.png', '').split('_var_')[0]
        clean_lookup[base_id] = path

    print(f"Found {len(clean_lookup)} base clean scenes.")

    print("2. Deduplicating hazy images (1 variation per scene)...")
    all_hazy_paths = glob.glob(os.path.join(hazy_dir, "**", "*.png"), recursive=True)

    hazy_list = []
    clean_list = []
    seen_base_ids = set()

    for path in sorted(all_hazy_paths):
        filename = os.path.basename(path)
        base_id = filename.replace('.png', '').split('_var_')[0]
        
        if base_id in clean_lookup and base_id not in seen_base_ids:
            seen_base_ids.add(base_id)
            hazy_list.append(path)
            clean_list.append(clean_lookup[base_id])

    print(f"Filtered down to {len(hazy_list)} UNIQUE, genuine scene pairs.")

    data_dict = {"hazy": hazy_list, "clear": clean_list}
    hf_dataset = Dataset.from_dict(data_dict)
    hf_dataset = hf_dataset.cast_column("hazy", Image())
    hf_dataset = hf_dataset.cast_column("clear", Image())

    dataset_dict = DatasetDict({"train": hf_dataset})

    print(f"3. Uploading clean dataset to Hugging Face repo: {hf_repo}...")
    dataset_dict.push_to_hub(hf_repo)
    print("SUCCESS! Unique dataset is live on Hugging Face!")

if __name__ == "__main__":
    prepare_and_upload_clean_dataset()