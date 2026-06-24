import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO

class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(6, 3, 7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(12, 3, 3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.relu(self.b1(self.conv1(x)))
        x2 = self.relu(self.b1(self.conv2(x1)))
        x5 = self.relu(self.conv5(torch.cat((x1, x2, self.relu(self.b1(self.conv3(torch.cat((x1, x2), 1)))) , self.relu(self.b1(self.conv4(torch.cat((x2, self.relu(self.b1(self.conv3(torch.cat((x1, x2), 1))))), 1))))), 1)))
        out = x5 * x - x5 + 1.0
        return torch.clamp(out, 0.0, 1.0)

class PixelAttention(nn.Module):
    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels // 8, 1), nn.ReLU(inplace=True), nn.Conv2d(channels // 8, 1, 1), nn.Sigmoid())
    def forward(self, x): return x * self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels // 8, 1), nn.ReLU(inplace=True), nn.Conv2d(channels // 8, channels, 1), nn.Sigmoid())
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
        self.fusion = nn.Sequential(nn.Conv2d(64 * blocks, 64, 1), nn.ReLU(inplace=True))
        self.ca = ChannelAttention(64)
        self.pa = PixelAttention(64)
        self.g2 = nn.Conv2d(64, 64, 3, padding=1)
        self.g3 = nn.Conv2d(64, out_channels, 3, padding=1)
    def forward(self, x):
        feat = self.g1(x)
        outputs = []
        out = feat
        for block in self.group:
            out = block(out)
            outputs.append(out)
        fused = self.pa(self.ca(self.fusion(torch.cat(outputs, dim=1))))
        return torch.clamp(self.g3(self.g2(fused) + feat) + x, 0.0, 1.0)


def execute_comparative_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("pipeline_outputs", exist_ok=True)
    
    print("\n==================================================")
    print("[INFO] Loading Framework Weights Into Memory...")
    
    ffa_model = FFANet(blocks=3).to(device)
    if os.path.exists("weights/ffanet_scratch.pth"):
        ffa_model.load_state_dict(torch.load("weights/ffanet_scratch.pth", map_location=device, weights_only=True))
    ffa_model.eval()

    aod_model = AODNet().to(device)
    if os.path.exists("weights/aodnet_optimized.pth"):
        aod_model.load_state_dict(torch.load("weights/aodnet_optimized.pth", map_location=device, weights_only=True))
    aod_model.eval()

    yolo_weight_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(yolo_weight_path):
        yolo_weight_path = "runs/detect/train-2/weights/best.pt"
    if not os.path.exists(yolo_weight_path):
        yolo_weight_path = "yolov8n.pt"
        
    print(f"[INFO] Initializing detector weights from: {yolo_weight_path}")
    detector = YOLO(yolo_weight_path)

    img_dir = "yolo_images"
    test_images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not test_images:
        print("[ERROR] No target cockpit frames found in yolo_images folder.")
        return
        
    sample_img_name = test_images[0]
    sample_path = os.path.join(img_dir, sample_img_name)
    print(f"[INFO] Extracting sample matrix from: {sample_path}")
    
    raw_cv = cv2.imread(sample_path)
    resized_cv = cv2.resize(raw_cv, (256, 256))
    clean_norm = resized_cv / 255.0
    
    beta = 0.12
    tx = np.exp(-beta * 15.0)
    foggy_norm = clean_norm * tx + 0.8 * (1.0 - tx)
    
    foggy_tensor = torch.from_numpy(foggy_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        aod_out = aod_model(foggy_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        ffa_out = ffa_model(foggy_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        
    foggy_img_out = np.ascontiguousarray((foggy_norm * 255.0).astype(np.uint8))
    aod_img_out = np.ascontiguousarray(aod_out.astype(np.uint8))
    ffa_img_out = np.ascontiguousarray(ffa_out.astype(np.uint8))

    print("[INFO] Evaluating Detector confidence profiles across modules...")
    res_foggy = detector(foggy_img_out, verbose=False)[0].plot()
    res_aod = detector(aod_img_out, verbose=False)[0].plot()
    res_ffa = detector(ffa_img_out, verbose=False)[0].plot()

    top_label_row = np.zeros((40, res_foggy.shape[1] * 3, 3), dtype=np.uint8)
    cv2.putText(top_label_row, "Path A: Raw Fog + YOLO", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(top_label_row, "Path B: AOD-Net + YOLO", (res_foggy.shape[1] + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(top_label_row, "Path C: FFA-Net + YOLO", (res_foggy.shape[1] * 2 + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    image_strip = np.hstack((res_foggy, res_aod, res_ffa))
    final_output_figure = np.vstack((top_label_row, image_strip))

    output_plot_path = "pipeline_outputs/thesis_comparison_strip.png"
    cv2.imwrite(output_plot_path, final_output_figure)
    
    print("\n==================================================")
    print(" [SUCCESS] CORE SELECTION EVALUATION COMPLETE!     ")
    print(f" Saved Comparison Plot: {output_plot_path} ")
    print("==================================================\n")

if __name__ == "__main__":
    execute_comparative_analysis()