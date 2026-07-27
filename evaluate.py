import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models import get_model

def evaluate_model(model_name, weight_path, num_samples=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = load_dataset("NeuroPropel/CockpitAI_dehaze_clean", split="train", streaming=True)
    
    psnr_scores = []
    ssim_scores = []

    print(f"\n--> Evaluating '{model_name}' on {num_samples} samples...")
    
    with torch.no_grad():
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            hazy_t = transform(item['hazy'].convert("RGB")).unsqueeze(0).to(device)
            clear_t = transform(item['clear'].convert("RGB")).unsqueeze(0).to(device)

            pred_t = model(hazy_t)

            # Convert tensors to numpy array image format [H, W, C]
            pred_img = pred_t.squeeze(0).cpu().permute(1, 2, 0).numpy()
            clear_img = clear_t.squeeze(0).cpu().permute(1, 2, 0).numpy()

            p = psnr(clear_img, pred_img, data_range=1.0)
            s = ssim(clear_img, pred_img, channel_axis=2, data_range=1.0)

            psnr_scores.append(p)
            ssim_scores.append(s)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print(f"=== Results for {model_name} ===")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

if __name__ == "__main__":
    # Example evaluation usage
    evaluate_model("aodnet", "model_aodnet.pth")