import os
import torch
import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from model import AODNet
# If AdverseFogSimulator is defined inline or in another module, make sure this matches:
from dataset_prep_large import AdverseFogSimulator

def mass_generate_dataset(clean_dir, output_dir):
    categories = ['cat1', 'cat2', 'cat3a']
    for cat in categories:
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ground_truth'), exist_ok=True)

    clean_images = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if os.path.exists(os.path.join(output_dir, 'cat3a')) and len(os.listdir(os.path.join(output_dir, 'cat3a'))) == len(clean_images):
        print(f"[INFO] Dataset of {len(clean_images)} images already exists. Skipping generation.")
        return

    print(f"\n[INFO] Found {len(clean_images)} images. Mass-generating fog profiles...")
    for img_name in clean_images:
        img_path = os.path.join(clean_dir, img_name)
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_dir, 'ground_truth', img_name), img)
        for cat in categories:
            simulator = AdverseFogSimulator(img_path)
            foggy_img = simulator.generate_fog(thickness_level=cat)
            cv2.imwrite(os.path.join(output_dir, cat, img_name), foggy_img)
    print(f"[SUCCESS] Full dataset generated inside: '{output_dir}/'\n")


def calculate_single_score(image_path, thickness_level='cat3a', model=None, device=None):
    simulator = AdverseFogSimulator(image_path)
    foggy_bgr = simulator.generate_fog(thickness_level=thickness_level)
    clean_bgr = simulator.img
    
    foggy_rgb = cv2.cvtColor(foggy_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Matched to training resolution
        transforms.ToTensor(),
    ])
    input_tensor = transform(foggy_rgb).unsqueeze(0).to(device)

    for _ in range(5): _ = model(input_tensor)
    if device.type == 'cuda': torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu()
    if device.type == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    fps_val = 1.0 / (end_time - start_time)
    
    clean_eval = cv2.resize(clean_bgr, (256, 256))
    foggy_eval = cv2.resize(foggy_bgr, (256, 256))
    
    output_np = output_tensor.permute(1, 2, 0).numpy()
    output_eval = (np.clip(output_np, 0.0, 1.0) * 255).astype(np.uint8)
    output_eval = cv2.cvtColor(output_eval, cv2.COLOR_RGB2BGR)

    psnr_val = psnr(clean_eval, output_eval, data_range=255)
    ssim_val = ssim(clean_eval, output_eval, channel_axis=2, data_range=255)

    print("=========================================")
    print(f"      SINGLE IMAGE BENCHMARK ({thickness_level.upper()})")
    print("=========================================")
    print(f" PSNR : {psnr_val:.2f} dB  |  SSIM : {ssim_val:.4f}")
    print(f" Latency : {latency_ms:.2f} ms |  FPS  : {fps_val:.1f}")
    print("=========================================\n")


def calculate_dataset_averages_and_save_visuals(fog_dir, gt_dir, visual_output_dir, model=None, device=None):
    images = [f for f in os.listdir(fog_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:20] # Test top 20 for speed tonight
    
    os.makedirs(visual_output_dir, exist_ok=True)
    
    print(f"[INFO] Evaluating averages and saving side-by-side results for {len(images)} sample validation images...")
    psnr_list, ssim_list = [], []
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor()])

    for img_name in images:
        foggy_bgr = cv2.imread(os.path.join(fog_dir, img_name))
        clean_bgr = cv2.imread(os.path.join(gt_dir, img_name))
        
        input_tensor = transform(cv2.cvtColor(foggy_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor).squeeze(0).cpu()
            
        clean_eval = cv2.resize(clean_bgr, (256, 256))
        foggy_eval = cv2.resize(foggy_bgr, (256, 256))
        
        output_np = output_tensor.permute(1, 2, 0).numpy()
        output_eval = (np.clip(output_np, 0.0, 1.0) * 255).astype(np.uint8)
        output_eval = cv2.cvtColor(output_eval, cv2.COLOR_RGB2BGR)

        psnr_list.append(psnr(clean_eval, output_eval, data_range=255))
        ssim_list.append(ssim(clean_eval, output_eval, channel_axis=2, data_range=255))

        comparison_grid = np.hstack((foggy_eval, output_eval, clean_eval))
        cv2.imwrite(os.path.join(visual_output_dir, f"result_{img_name}"), comparison_grid)

    print(f"[SUCCESS] Visual panel verification exported inside: '{visual_output_dir}/'")
    print("=========================================")
    print("   AOD-NET TRUE DATASET BULK AVERAGES   ")
    print("=========================================")
    print(f" Total Images Processed : {len(images)}")
    print(f" Grand Average PSNR     : {np.mean(psnr_list):.2f} dB")
    print(f" Grand Average SSIM     : {np.mean(ssim_list):.4f}")
    print("=========================================\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_model = AODNet().to(device)
    
    # Linked directly to your fresh survival weights
    shared_model.load_state_dict(torch.load("weights/aodnet_pilot_success.pt", map_location=device))
    shared_model.eval()

    # Absolute locations from your system configuration
    HAZY_DATA_DIR = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\hazy"
    CLEAN_DATA_DIR = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset\train\clean"
    OUTPUT_VISUALS = r"C:\Users\ACER\Desktop\Cockpit_AI\output_results"
    
    # Run single benchmark using the first available image in the hazy directory
    sample_image_name = [f for f in os.listdir(CLEAN_DATA_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))][0]
    test_img_path = os.path.join(CLEAN_DATA_DIR, sample_image_name)
    
    calculate_single_score(test_img_path, thickness_level='cat3a', model=shared_model, device=device)
    
    # Run verification matrix processing
    calculate_dataset_averages_and_save_visuals(HAZY_DATA_DIR, CLEAN_DATA_DIR, OUTPUT_VISUALS, model=shared_model, device=device)