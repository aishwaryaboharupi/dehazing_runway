import os
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models import get_model


def load_checkpoint_weights(model, weight_path, device):
    """Safely loads model weights handling full resume dicts or raw state dicts."""
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    device = torch.device("cpu")
    os.makedirs("figures", exist_ok=True)

    # 1. Pick a random sample from local folder
    sample_files = [f for f in os.listdir("test_samples/hazy") if f.endswith(".png")]
    if not sample_files:
        print("--> Error: No local images found. Did you run download_samples.py?")
        return

    chosen_file = random.choice(sample_files)
    sample_id = chosen_file.replace(".png", "")
    print(f"--> Using local test image: {chosen_file}")

    hazy_img_pil = Image.open(f"test_samples/hazy/{chosen_file}").resize((256, 256))
    clear_img_pil = Image.open(f"test_samples/clear/{chosen_file}").resize((256, 256))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    hazy_t = transform(hazy_img_pil).unsqueeze(0).to(device)

    models_config = [
        ("AOD-Net", "aodnet", "checkpoints/aodnet_resume_epoch_30.pth"),
        ("FFA-Net (Saturated)", "ffanet", "checkpoints/ffanet_resume_epoch_30.pth"),
        ("Mamba", "mamba", "checkpoints/mamba_resume_epoch_30.pth"),
        ("DehazeTransformer", "transformer", "checkpoints/transformer_resume_epoch_30.pth"),
    ]

    outputs = {}
    for display_name, model_key, weight_path in models_config:
        if os.path.exists(weight_path):
            print(f"--> Generating prediction for {display_name}...")
            model = get_model(model_key).to(device)
            model = load_checkpoint_weights(model, weight_path, device)
            
            with torch.no_grad():
                pred_t = model(hazy_t)
                pred_t = torch.clamp(pred_t, 0.0, 1.0)
                pred_img = pred_t.squeeze(0).permute(1, 2, 0).numpy()
                outputs[display_name] = pred_img
        else:
            print(f"--> Warning: Weight file '{weight_path}' not found locally. Skipping {display_name}.")

    print("--> Plotting comparative figure...")
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    
    # 1. Hazy Input
    axes[0].imshow(hazy_img_pil)
    axes[0].set_title("Hazy Input", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    # 2-5. Model Predictions
    for idx, (display_name, _, _) in enumerate(models_config, start=1):
        if display_name in outputs:
            axes[idx].imshow(outputs[display_name])
            axes[idx].set_title(display_name, fontsize=11, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, "Missing Weight", ha='center', va='center')
        axes[idx].axis("off")

    # 6. Ground Truth
    axes[5].imshow(clear_img_pil)
    axes[5].set_title("Ground Truth", fontsize=11, fontweight='bold')
    axes[5].axis("off")

    plt.tight_layout()
    output_path = f"figures/comparison_{sample_id}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"--> Done! Figure saved to: '{output_path}'")


if __name__ == "__main__":
    main()