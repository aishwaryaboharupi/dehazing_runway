import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import sys


class PixelAttention(nn.Module):
    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.conv(self.gap(x))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels)
        self.pa = PixelAttention(channels)
    def forward(self, x):
        return x + self.pa(self.ca(self.conv2(self.relu(self.conv1(x)))))

class FFANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, blocks=3):
        super(FFANet, self).__init__()
        self.g1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.group = nn.ModuleList([ResidualAttentionBlock(64) for _ in range(blocks)])
        self.fusion = nn.Sequential(nn.Conv2d(64 * blocks, 64, 1, padding=0), nn.ReLU(inplace=True))
        self.ca = ChannelAttention(64)
        self.pa = PixelAttention(64)
        self.g2 = nn.Conv2d(64, 64, 3, padding=1)
        self.g3 = nn.Conv2d(64, out_channels, 3, padding=1)
    def forward(self, x):
        feat = self.g1(x)
        group_outputs = []
        out = feat
        for block in self.group:
            out = block(out)
            group_outputs.append(out)
        fused = self.fusion(torch.cat(group_outputs, dim=1))
        fused = self.pa(self.ca(fused))
        return torch.clamp(self.g3(self.g2(fused) + feat) + x, 0.0, 1.0)


def calculate_ssim(img1, img2):
    """Calculates structural similarity index baseline between two images."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def run_metrics_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==================================================")
    print(f"[INFO] Initializing Quantitative Evaluation on: {device}")
    print("==================================================")

    model = FFANet(blocks=3).to(device)
    weight_path = "weights/ffanet_scratch.pth"
    if not os.path.exists(weight_path):
        print(f"[ERROR] Weights file missing at {weight_path}. Run training first.")
        return
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    img_dir = "clean_images"
    if not os.path.exists(img_dir):
        print(f"[ERROR] Folder '{img_dir}' does not exist.")
        return
        
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_imgs = len(image_files)
    
    if total_imgs == 0:
        print("[ERROR] No images found inside clean_images folder.")
        return

    mse_list, psnr_list, ssim_list = [], [], []

    print(f"[INFO] Evaluating {total_imgs} images through the FFA network...")
    
    with torch.no_grad():
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(img_dir, img_name)
            
            raw_cv = cv2.imread(img_path)
            clean_res = cv2.resize(raw_cv, (256, 256))
            clean_norm = clean_res / 255.0
            
            beta = 0.12
            tx = np.exp(-beta * 15.0)
            foggy_norm = clean_norm * tx + 0.8 * (1.0 - tx)
            
            foggy_tensor = torch.from_numpy(foggy_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)
            output_tensor = model(foggy_tensor)
            
            restored_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            restored_img = restored_np.astype(np.uint8)
            ground_truth_img = (clean_norm * 255.0).astype(np.uint8)
            
            mse = np.mean((ground_truth_img - restored_img) ** 2)
            psnr = cv2.PSNR(ground_truth_img, restored_img)
            ssim = calculate_ssim(ground_truth_img, restored_img)
            
            mse_list.append(mse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            sys.stdout.write(f"\r Processing Progress: {i+1}/{total_imgs} images analyzed...")
            sys.stdout.flush()

    final_mse = np.mean(mse_list)
    final_psnr = np.mean(psnr_list)
    final_ssim = np.mean(ssim_list)

    print("\n\n==================================================")
    print("      FFA-NET QUANTITATIVE METRICS RESULTS        ")
    print("==================================================")
    print(f" Average Mean Squared Error (MSE):     {final_mse:.6f}")
    print(f" Average Peak Signal-to-Noise (PSNR):  {final_psnr:.2f} dB")
    print(f" Average Structural Similarity (SSIM): {final_ssim:.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_metrics_evaluation()
    