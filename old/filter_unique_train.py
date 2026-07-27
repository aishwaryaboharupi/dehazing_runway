import os
import glob
import shutil

# =========================================================
# PATHS CONFIGURATION
# =========================================================
SOURCE_TRAIN_HAZY = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\hazy"
SOURCE_TRAIN_CLEAN = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\clean"

OUTPUT_DIR = r"C:\Users\ACER\Desktop\Cockpit_AI\filtered_train_dataset"
OUTPUT_HAZY = os.path.join(OUTPUT_DIR, "train", "hazy")
OUTPUT_CLEAN = os.path.join(OUTPUT_DIR, "train", "clean")

os.makedirs(OUTPUT_HAZY, exist_ok=True)
os.makedirs(OUTPUT_CLEAN, exist_ok=True)

# =========================================================
# FILTERING PROCESS
# =========================================================
print("Scanning all 12 part folders for hazy images...")

# Recursively search for all images in hazy subfolders
hazy_image_paths = glob.glob(os.path.join(SOURCE_TRAIN_HAZY, "**", "*.png"), recursive=True) + \
                   glob.glob(os.path.join(SOURCE_TRAIN_HAZY, "**", "*.jpg"), recursive=True)

print(f"Total hazy images found across all parts: {len(hazy_image_paths)}")

seen_base_names = set()
copied_count = 0

print("Filtering to 1 unique hazy image per runway base scene...")

for hazy_path in hazy_image_paths:
    filename = os.path.basename(hazy_path)
    
    # Extract original base name (stripping category/fog tags)
    # Example: 'EDDF07L1_cat3.png' -> 'EDDF07L1'
    base_name = filename.split('_cat')[0].replace('_clean', '').replace('.png', '').replace('.jpg', '')

    # Skip if we already picked a hazy version of this runway scene
    if base_name in seen_base_names:
        continue

    # Find the corresponding clean ground truth image
    # Checks both part folders and root clean folder
    clean_filename = f"{base_name}.png"
    clean_matches = glob.glob(os.path.join(SOURCE_TRAIN_CLEAN, "**", clean_filename), recursive=True)
    
    if not clean_matches:
        # Try matching exact filename if named identically
        clean_matches = glob.glob(os.path.join(SOURCE_TRAIN_CLEAN, "**", filename), recursive=True)

    if clean_matches:
        clean_path = clean_matches[0]
        
        # Copy the unique pair to our filtered output directory
        shutil.copy(hazy_path, os.path.join(OUTPUT_HAZY, filename))
        shutil.copy(clean_path, os.path.join(OUTPUT_CLEAN, filename))
        
        seen_base_names.add(base_name)
        copied_count += 1

print("\n=========================================================")
print(f" SUCCESS! Filtered dataset created.")
print(f" Total Unique Image Pairs Kept: {copied_count}")
print(f" Location: {OUTPUT_DIR}")
print("=========================================================")