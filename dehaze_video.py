import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import imageio
from models import get_model


def load_checkpoint_weights(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    device = torch.device("cpu")
    
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        print(f"--> Error: '{video_path}' not found in project directory.")
        return

    model_name = "transformer"
    weight_path = "checkpoints/transformer_resume_epoch_30.pth"

    if not os.path.exists(weight_path):
        print(f"--> Error: Weight file '{weight_path}' not found.")
        return

    print(f"--> Loading trained DehazeTransformer from '{weight_path}'...")
    model = get_model(model_name).to(device)
    model = load_checkpoint_weights(model, weight_path, device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"--> Error: OpenCV could not open '{video_path}'.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 60:
        fps = 30

    max_frames = 180  # ~6 seconds
    width, height = 256, 256

    transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor()
    ])

    output_filename = "cockpit_dehazing_demo.mp4"
    writer = imageio.get_writer(output_filename, fps=fps, codec='libx264', quality=8)

    print(f"--> Processing {max_frames} frames (Direct RGB)...")
    processed_count = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret or processed_count >= max_frames:
                break
            
            # 1. Standard OpenCV BGR -> RGB PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).convert("RGB")
            
            # 2. PyTorch transform
            input_t = transform(pil_img).unsqueeze(0).to(device)
            
            # 3. Predict
            pred_t = model(input_t)
            
            # 4. DIRECT PREDICTION CONVERSION (No residual math, no histogram matching)
            # If the model output is inverted, (1 - pred_t) restores normal lightness
            pred_img = pred_t.squeeze(0).cpu().permute(1, 2, 0).numpy()
            
            # Simple inversion check: if mean is very low (< 0.3), invert lightness
            if pred_img.mean() < 0.35:
                pred_img = 1.0 - pred_img
                
            pred_img = np.clip(pred_img, 0.0, 1.0)
            dehazed_rgb = (pred_img * 255.0).astype(np.uint8)
            
            # 5. Raw Hazy Frame
            hazy_rgb = cv2.resize(frame_rgb, (width, height))
            
            # 6. Side-by-Side (RGB)
            combined_frame = np.hstack((hazy_rgb, dehazed_rgb))
            
            # 7. Labels
            cv2.putText(combined_frame, "RAW HAZY FEED", (10, 22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(combined_frame, "DEHAZED (TRANSFORMER)", (266, 22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            writer.append_data(combined_frame)
            processed_count += 1
            
            if processed_count % 30 == 0:
                print(f"    Processed {processed_count}/{max_frames} frames...")

    cap.release()
    writer.close()
    print(f"\n--> SUCCESS! Saved video: '{output_filename}'")


if __name__ == "__main__":
    main()