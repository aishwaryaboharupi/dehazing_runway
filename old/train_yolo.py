import os
import cv2
import numpy as np
from ultralytics import YOLO

def build_and_train_from_scratch():
    img_dir = "yolo_images"
    mask_source_dir = "label"
    
    # YOLO looks for a folder named 'labels' parallel to 'yolo_images'
    target_label_dir = "yolo_labels"
    os.makedirs(target_label_dir, exist_ok=True)
    
    print("\n==================================================")
    print("[INFO] Step 1: Mapping coordinates for your 97 target images...")
    
    # Cache all available masks by their airport prefix prefix (e.g. 'LFPG09R1')
    mask_pool = {}
    if os.path.exists(mask_source_dir):
        for f in os.listdir(mask_source_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')) and '_' in f:
                prefix = f.split('_')[0]
                if prefix not in mask_pool:
                    mask_pool[prefix] = os.path.join(mask_source_dir, f)

    paired_count = 0
    
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_name)[0]
            prefix = img_name.split('_')[0] if '_' in img_name else ""
            
            # 1. Try exact filename match first
            mask_path = os.path.join(mask_source_dir, f"{base_name}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_source_dir, f"{base_name}.jpg")
            
            # 2. Fallback to any mask sharing the same airport prefix layout
            if not os.path.exists(mask_path) and prefix in mask_pool:
                mask_path = mask_pool[prefix]
                
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                h, w = mask.shape[:2]
                txt_path = os.path.join(target_label_dir, f"{base_name}.txt")
                
                with open(txt_path, "w") as f:
                    for contour in contours:
                        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
                        x_center = (x + bbox_w / 2.0) / w
                        y_center = (y + bbox_h / 2.0) / h
                        yolo_w = bbox_w / w
                        yolo_h = bbox_h / h
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
                paired_count += 1

    print(f"[SUCCESS] Paired and built coordinates for {paired_count} images.")
    
    if paired_count == 0:
        print("[ERROR] No matching or prefix-compatible shapes found. Please verify folder contents.")
        return

    # 2. Generate the YAML configuration file
    print("[INFO] Step 2: Generating runtime configuration scratch_dataset.yaml...")
    yaml_content = f"""
path: {os.path.abspath('.')}
train: {img_dir}
val: {img_dir}
names:
  0: Runway
"""
    with open("scratch_dataset.yaml", "w") as f:
        f.write(yaml_content.strip())

    # 3. Initialize and train YOLOv8 from scratch
    print("[INFO] Step 3: Loading baseline yolov8n.pt framework...")
    model = YOLO("yolov8n.pt")
    
    print("[INFO] Step 4: Booting training execution loop...")
    model.train(
        data="scratch_dataset.yaml", 
        epochs=25, 
        imgsz=640
    )
    
    # 4. Clean up temporary configuration file
    if os.path.exists("scratch_dataset.yaml"):
        os.remove("scratch_dataset.yaml")
    print("\n==================================================")
    print("  SUCCESS: RUNWAY TRAINING PIPELINE COMPLETE!     ")
    print("==================================================\n")

if __name__ == "__main__":
    build_and_train_from_scratch()