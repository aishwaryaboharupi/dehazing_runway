import os
import cv2
import numpy as np
import glob

# =========================================================
# 1. SET YOUR DESKTOP PATH HERE
# Replace the path below with your actual Desktop folder path
# =========================================================
DESKTOP_INPUT_DIR = r"C:\Users\ACER\Desktop\test"  
OUTPUT_DIR = "./lard_test_dataset"

# Output folders inside your project
HAZY_OUT = os.path.join(OUTPUT_DIR, "hazy")
CLEAN_OUT = os.path.join(OUTPUT_DIR, "clean")

os.makedirs(HAZY_OUT, exist_ok=True)
os.makedirs(CLEAN_OUT, exist_ok=True)

# 2. Atmospheric Scattering Fog Generator
def apply_haze(image, beta, A=0.85):
    h, w, c = image.shape
    size = max(h, w)
    center_x, center_y = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    depth = np.sqrt((x - center_x)**2 + (y - center_y)**2) / size
    depth = depth[:, :, np.newaxis]

    transmission = np.exp(-beta * depth)
    transmission = np.clip(transmission, 0.1, 1.0)

    img_float = image.astype(np.float32) / 255.0
    hazy = img_float * transmission + A * (1.0 - transmission)
    return np.clip(hazy * 255.0, 0, 255).astype(np.uint8)

# 3. Limit to exactly 75 images from Desktop (75 x 2 haze levels = 150 total test images)
LIMIT_IMAGES = 250

search_pattern = os.path.join(DESKTOP_INPUT_DIR, "*.*")
all_files = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(all_files)} total images on Desktop.")
print(f"Selecting the first {min(LIMIT_IMAGES, len(all_files))} images for Cat 3 & Cat 4 test set generation...")

count = 0
for img_path in all_files[:LIMIT_IMAGES]:
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    img = cv2.imread(img_path)
    
    if img is None:
        continue

    # Generate Cat 3 (Medium-Heavy Fog) and Cat 4 (Dense Fog)
    cat3_img = apply_haze(img, beta=2.0, A=0.85)
    cat4_img = apply_haze(img, beta=3.5, A=0.90)

    # Save Clean target
    cv2.imwrite(os.path.join(CLEAN_OUT, f"{name}_clean{ext}"), img)
    cv2.imwrite(os.path.join(CLEAN_OUT, f"{name}_cat3_clean{ext}"), img)
    cv2.imwrite(os.path.join(CLEAN_OUT, f"{name}_cat4_clean{ext}"), img)

    # Save Hazy inputs
    cv2.imwrite(os.path.join(HAZY_OUT, f"{name}_cat3{ext}"), cat3_img)
    cv2.imwrite(os.path.join(HAZY_OUT, f"{name}_cat4{ext}"), cat4_img)

    count += 1
    print(f"[{count}/{min(LIMIT_IMAGES, len(all_files))}] Converted: {filename} -> Cat 3 & Cat 4 created!")

print(f"\nSuccess! Created {count * 2} test pairs in '{OUTPUT_DIR}'. Ready for Hugging Face upload!")