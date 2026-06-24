import cv2
import numpy as np
import os

class AdverseFogSimulator:
    def __init__(self, raw_image_path):
        self.img = cv2.imread(raw_image_path)
        if self.img is None:
            raise FileNotFoundError(f"Could not load raw image at: {raw_image_path}")
        self.height, self.width, self.channels = self.img.shape

    def _make_organic_noise(self, scale_coarse=40, scale_fine=12, blur_coarse=151, blur_fine=51):
        """
        Multi-octave Perlin-like noise via blurred random fields.
        Returns a [H, W] map in [0, 1].
        """
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
        """
        Physically-based fog using Koschmieder's law.
        Transmission is spatially modulated by organic noise so fog looks
        volumetric and non-uniform, not a flat white sheet.

        thickness_level:
            'cat1'  — CAT I,   RVR ~550m,  max_t=0.30
            'cat2'  — CAT II,  RVR ~300m,  max_t=0.12
            'cat3a' — CAT IIIa,RVR ~200m,  max_t=0.06
            'cat3b' — CAT IIIb,RVR  ~75m,  max_t=0.025
            'cat3c' — CAT IIIc,RVR  ~25m,  max_t=0.008
            'heavy' — alias for cat3b
        """
        presets = {
            'cat1':  (0.30, 0.05),   # (t_max, noise_strength)
            'cat2':  (0.12, 0.08),
            'cat3a': (0.06, 0.10),
            'cat3b': (0.025, 0.14),
            'cat3c': (0.008, 0.18),
            'heavy': (0.025, 0.14),
        }
        t_max, noise_strength = presets.get(thickness_level, (0.025, 0.14))

        img_normalized = self.img.astype(np.float32) / 255.0

        y = np.linspace(1.0, 0.55, self.height).reshape(self.height, 1)
        depth_map = np.tile(y, (1, self.width))  # [H, W]

        fog_noise = self._make_organic_noise()  # [H, W] in [0,1]

       
        beta = -np.log(max(t_max, 1e-6))
        base_t = np.exp(-beta * depth_map)  # [H, W]


        modulator = 1.0 + noise_strength * (fog_noise * 2.0 - 1.0)
        noisy_t = base_t * modulator

        
        noisy_t = np.clip(noisy_t, 0.001, t_max * 1.8)
        transmission_map = noisy_t[:, :, np.newaxis]   

       
        airlight = np.array([A * 0.97, A * 0.99, A * 1.00],    
                             dtype=np.float32).reshape(1, 1, 3)

        
        fog_wall_variation = 1.0 - 0.06 * fog_noise[:, :, np.newaxis]
        airlight_spatial = np.clip(airlight * fog_wall_variation, 0, 1)

        
        foggy = img_normalized * transmission_map + airlight_spatial * (1.0 - transmission_map)

        return (np.clip(foggy, 0.0, 1.0) * 255).astype(np.uint8)


if __name__ == "__main__":
    os.makedirs("foggy_images", exist_ok=True)

    input_file = os.path.join("clean_images", "runway_1.png")
    simulator  = AdverseFogSimulator(input_file)

    levels = ['cat1', 'cat2', 'cat3a', 'cat3b', 'cat3c']
    for lvl in levels:
        out_path = os.path.join("foggy_images", f"runway_{lvl}.png")
        result   = simulator.generate_fog(thickness_level=lvl)
        cv2.imwrite(out_path, result)
        print(f"Saved {lvl} → {out_path}")