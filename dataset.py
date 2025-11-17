# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class CrowdDataset(Dataset):
    def __init__(self, img_dir, density_dir, transform=None):
        self.img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        self.density_paths = [os.path.join(density_dir, f.replace('.jpg', '.npy')) 
                              for f in sorted(os.listdir(img_dir))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        density = np.load(self.density_paths[idx])
        # Apply image transform (e.g., resize, normalize)
        if self.transform:
            img = self.transform(img)
            # img is a tensor with shape [C, H, W]
            _, H, W = img.shape
            # CSRNet frontend reduces spatial resolution. For this model configuration
            # with VGG features sliced up to conv3, the downscale factor is 4
            downscale = 4
            new_h = max(1, H // downscale)
            new_w = max(1, W // downscale)
            # Original density map size
            orig_h, orig_w = density.shape
            # Resize density to match prediction spatial size (cv2 resize expects (w,h))
            density_resized = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # Preserve total count: scale by area ratio
            if new_h * new_w > 0:
                scale = (orig_h * orig_w) / (new_h * new_w)
                density_resized = density_resized * scale
            density = density_resized
        else:
            # No transform: convert density as-is
            pass

        density = torch.from_numpy(density).unsqueeze(0).float()
        return img, density