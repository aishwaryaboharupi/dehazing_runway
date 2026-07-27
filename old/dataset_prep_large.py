import os
import cv2
import numpy as np
import random

class AdverseFogSimulator:
    def __init__(self, raw_image_path):
        self.img = cv2.imread(raw_image_path)
        if self.img is None:
            raise FileNotFoundError(f"Could not load raw image at: {raw_image_path}")
        self.height, self.width, self.channels = self.img.shape

    def _make_organic_noise(self, scale_coarse=40, scale_fine=12, blur_coarse=151, blur_fine=51):
        n1 = np.random.rand(self.height // scale_coarse + 2,
                            self.width // scale_coarse + 2).astype(np.float32)
        n1 = cv2.resize(n1, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        n1 = cv2.GaussianBlur(n1, (blur_coarse | 1, blur_coarse | 1), 0)

        n2 = np.random.rand(self.height // 20 + 2,
                            self.width // 20 + 2).astype(np.float32)
        n2 = cv2.resize(n2, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        n2 = cv2.GaussianBlur(n2, (71, 71), 0)

        n3 = np.random.rand(self.height // scale_fine + 2,
                            self.width // scale_fine + 2).astype(np.float32)
        n3 = cv2.resize(n3, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        n3 = cv2.GaussianBlur(n3, (31, 31), 0)

        combined = n1 * 0.55 + n2 * 0.30 + n3 * 0.15
        combined -= combined.min()
        combined /= (combined.max() + 1e-8)
        return combined

    def generate_fog(self, thickness_level='heavy', A=0.98):
        presets = {
            'cat3a': (0.06, 0.10),
            'cat3b': (0.025, 0.14),
            'cat3c': (0.008, 0.18),
            'heavy': (0.025, 0.14),
        }
        t_max, noise_strength = presets.get(thickness_level, (0.025, 0.14))

        img_normalized = self.img.astype(np.float32) / 255.0
        y = np.linspace(1.0, 0.55, self.height).reshape(self.height, 1)
        depth_map = np.tile(y, (1, self.width))
        fog_noise = self._make_organic_noise()

        beta = -np.log(max(t_max, 1e-6))
        base_t = np.exp(-beta * depth_map)

        modulator = 1.0 + noise_strength * (fog_noise * 2.0 - 1.0)
        noisy_t = base_t * modulator
        noisy_t = np.clip(noisy_t, 0.001, t_max * 1.8)
        transmission_map = noisy_t[:, :, np.newaxis]   

        airlight = np.array([A * 0.97, A * 0.99, A * 1.00], dtype=np.float32).reshape(1, 1, 3)
        fog_wall_variation = 1.0 - 0.06 * fog_noise[:, :, np.newaxis]
        airlight_spatial = np.clip(airlight * fog_wall_variation, 0, 1)

        foggy = img_normalized * transmission_map + airlight_spatial * (1.0 - transmission_map)
        return (np.clip(foggy, 0.0, 1.0) * 255).astype(np.uint8)


def build_massive_dataset(clean_source_dir, output_base_dir, variations_per_image=15):
    hazy_out_dir = os.path.join(output_base_dir, "train", "hazy")
    clean_out_dir = os.path.join(output_base_dir, "train", "clean")
    
    os.makedirs(hazy_out_dir, exist_ok=True)
    os.makedirs(clean_out_dir, exist_ok=True)
    
    target_levels = ['cat3a', 'cat3b', 'cat3c']
    clean_files = [f for f in os.listdir(clean_source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Targeting Source Directory: {clean_source_dir}")
    print(f"Found {len(clean_files)} high-res frames. Generating {variations_per_image} variants per image...")
    
    total_counter = 0
    for file_name in clean_files:
        input_path = os.path.join(clean_source_dir, file_name)
        base_name, ext = os.path.splitext(file_name)
        
        try:
            simulator = AdverseFogSimulator(input_path)
            for i in range(variations_per_image):
                selected_level = random.choice(target_levels)
                random_A = random.uniform(0.93, 0.99)
                
                foggy_frame = simulator.generate_fog(thickness_level=selected_level, A=random_A)
                new_name = f"{base_name}_var_{i}{ext}"
                
                cv2.imwrite(os.path.join(hazy_out_dir, new_name), foggy_frame)
                cv2.imwrite(os.path.join(clean_out_dir, new_name), simulator.img)
                total_counter += 1
                
        except Exception as e:
            print(f"Skipping file {file_name}: {e}")
            
    print(f"\nGeneration complete! Total images built: {total_counter}")


if __name__ == "__main__":
    # Handshaking with your exact local folder paths
    CLEAN_SOURCE = r"C:\Users\ACER\Desktop\msc-sem2\Thesis\archive\1920x1080\1920x1080\train"
    OUTPUT_DATASET = r"C:\Users\ACER\Desktop\Cockpit_AI\large_dataset"
    
    build_massive_dataset(CLEAN_SOURCE, OUTPUT_DATASET, variations_per_image=15)