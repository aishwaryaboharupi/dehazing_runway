import os
import time
from huggingface_hub import HfApi, login, CommitOperationAdd

# =========================================================
# CONFIGURATION
# =========================================================
# Token is read from environment variable or set to placeholder for security
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
REPO_ID = "NeuroPropel/CockpitAI_dehaze_dataset"

SOURCE_TRAIN_HAZY = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\hazy"
SOURCE_TRAIN_CLEAN = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\clean"

BATCH_SIZE = 400  # 400 pairs = 800 files per commit

# =========================================================
# 1. MAP CLEAN IMAGES INTO MEMORY
# =========================================================
print("Indexing clean ground-truth images...")
clean_map = {}
for root, _, files in os.walk(SOURCE_TRAIN_CLEAN):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            clean_map[f] = os.path.join(root, f)

print(f"Mapped {len(clean_map)} clean files.")

# =========================================================
# 2. FILTER UNIQUE SCENES BY SPLITTING ON `_var_`
# =========================================================
print("Filtering unique runway scenes (removing _var_ fog duplicates)...")
seen_base_names = set()
unique_pairs = []

for root, _, files in os.walk(SOURCE_TRAIN_HAZY):
    for f in files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        base_name = f.split('_var_')[0].split('_cat')[0].replace('_clean', '').replace('.png', '').replace('.jpg', '')

        if base_name in seen_base_names:
            continue

        hazy_path = os.path.join(root, f)
        clean_path = clean_map.get(f"{base_name}.png") or clean_map.get(f"{base_name}.jpg") or clean_map.get(f)

        if clean_path and os.path.exists(clean_path):
            unique_pairs.append((hazy_path, clean_path, f))
            seen_base_names.add(base_name)

print(f"\nSUCCESS! Reduced dataset to {len(unique_pairs)} STRICTLY UNIQUE scene pairs!")

# =========================================================
# 3. STREAM UPLOAD IN PARTITIONED SUBFOLDERS
# =========================================================
print("\nLogging into Hugging Face...")
if HF_TOKEN != "YOUR_HF_TOKEN_HERE":
    login(token=HF_TOKEN)

api = HfApi()

operations = []
commit_count = 0

for idx, (hazy_p, clean_p, fname) in enumerate(unique_pairs, 1):
    part_folder = f"part_{((idx - 1) // 3000) + 1}"
    
    operations.append(CommitOperationAdd(
        path_in_repo=f"filtered_train/{part_folder}/hazy/{fname}", 
        path_or_fileobj=hazy_p
    ))
    operations.append(CommitOperationAdd(
        path_in_repo=f"filtered_train/{part_folder}/clean/{fname}", 
        path_or_fileobj=clean_p
    ))

    if len(operations) >= BATCH_SIZE * 2 or idx == len(unique_pairs):
        commit_count += 1
        print(f"\nSubmitting Batch #{commit_count} ({len(operations)} files)...")
        
        success = False
        while not success:
            try:
                api.create_commit(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Clean train batch #{commit_count}"
                )
                print(f"Batch #{commit_count} uploaded! [{idx}/{len(unique_pairs)} pairs]")
                success = True
            except Exception as e:
                print(f"Error encountered: {e}")
                print("Waiting 60 seconds before retrying...")
                time.sleep(60)

        operations = []

print("\n=========================================================")
print(" SUCCESS! Your clean, leak-free training dataset is live!")
print(f" Dataset Link: https://huggingface.co/datasets/{REPO_ID}")
print("=========================================================")