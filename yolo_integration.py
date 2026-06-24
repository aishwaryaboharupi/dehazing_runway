import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from model import AODNet
from torchvision import transforms

def run_dual_stage_pipeline(dataset_dir, output_visual_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Initializing Dual-Stage Vision Pipeline on device: {device}")
    
    # 1. Load your trained Dehazing Network (AOD-Net)
    dehaze_model = AODNet().to(device)
    dehaze_model.load_state_dict(torch.load("weights/aodnet_cockpit.pth", map_location=device, weights_only=True))
    dehaze_model.eval()
    print("[SUCCESS] Stage 1: AOD-Net Weights loaded securely.")
    
    # 2. Load the standard YOLOv8 Pre-trained Model 
    # (Downloads a small 6MB base file on your first run automatically)
    detector = YOLO("yolov8n.pt") 
    print("[SUCCESS] Stage 2: YOLOv8 Object Detector initialized.")
    
    # 3. Select a target image from your heavy fog (CAT3a) dataset folder
    fog_dir = os.path.join(dataset_dir, "cat3a")
    images = [f for f in os.listdir(fog_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"[ERROR] No images found inside '{fog_dir}'. Please verify your dataset path.")
        return
        
    target_img_name = images[0]  # Pick the first available image to verify
    foggy_image_path = os.path.join(fog_dir, target_img_name)
    
    # 4. Preprocess image for AOD-Net
    img_bgr = cv2.imread(foggy_image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    # 5. Execution - Stage 1: Strip the Fog
    with torch.no_grad():
        cleared_tensor = dehaze_model(input_tensor).squeeze(0).cpu()
    
    # Convert tensor back to OpenCV BGR format for YOLO
    cleared_np = cleared_tensor.permute(1, 2, 0).numpy()
    cleared_bgr = (np.clip(cleared_np, 0.0, 1.0) * 255).astype(np.uint8)
    cleared_bgr = cv2.cvtColor(cleared_bgr, cv2.COLOR_RGB2BGR)
    
    # 6. Execution - Stage 2: Detect Objects on Cleared Image
    print("[INFO] Passing defogged image frame into YOLOv8...")
    results = detector(cleared_bgr)
    
    # 7. Render bounding boxes and save visual proof
    annotated_frame = results[0].plot()
    
    # Build a visual comparison: Original Heavy Fog vs. Defogged + YOLO Detected
    foggy_resized = cv2.resize(img_bgr, (640, 480))
    pipeline_comparison = np.hstack((foggy_resized, annotated_frame))
    
    cv2.imwrite(output_visual_path, pipeline_comparison)
    
    print("\n==================================================")
    print("      INTEGRATED COCKPIT VISION PIPELINE SCORE     ")
    print("==================================================")
    print(f" Target Processed   : {target_img_name}")
    print(f" Status             : Complete Execution Success")
    print(f" Visual Proof Saved : '{output_visual_path}'")
    print("==================================================\n")

if __name__ == "__main__":
    run_dual_stage_pipeline(dataset_dir="processed_dataset", output_visual_path="yolo_pipeline_test.png")