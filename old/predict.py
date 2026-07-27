import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from old.model import AODNet
from old.dataset_prep import AdverseFogSimulator

def run_inference(image_path, output_dir="results", thickness_level='cat3a'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AODNet().to(device)
    weights_path = os.path.join("weights", "aodnet_cockpit.pth")
    
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}. Train the model first!")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  
    print("Loaded trained AOD-Net weights successfully.")

    simulator = AdverseFogSimulator(image_path)
    foggy_bgr = simulator.generate_fog(thickness_level=thickness_level)
    clean_bgr = simulator.img
    
    foggy_rgb = cv2.cvtColor(foggy_bgr, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])
    
    
    input_tensor = transform(foggy_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_tensor = output_tensor.squeeze(0).cpu() 
    output_np = output_tensor.permute(1, 2, 0).numpy() 
    output_np = (np.clip(output_np, 0.0, 1.0) * 255).astype(np.uint8) 
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR) 
    clean_resized = cv2.resize(clean_bgr, (640, 480))
    foggy_resized = cv2.resize(foggy_bgr, (640, 480))

    comparison_canvas = np.hstack((clean_resized, foggy_resized, output_bgr))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dehaze_comparison_{thickness_level}.png")
    cv2.imwrite(output_path, comparison_canvas)
    print(f"Saved side-by-side visual evaluation canvas at: {output_path}")

    window_name = "AOD-Net Evaluation (Clean vs Foggy vs AI Restored)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow(window_name, 1280, 360) 
    
    cv2.imshow(window_name, comparison_canvas)
if __name__ == "__main__":
    test_image = os.path.join("clean_images", "runway_1.png")
    run_inference(test_image, thickness_level='cat3a')