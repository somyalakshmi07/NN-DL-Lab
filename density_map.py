# density_map.py
import os
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import h5py

def generate_density_map(img_path, points, sigma=4):
    img = cv2.imread(img_path, 0)
    h, w = img.shape
    density = np.zeros((h, w), dtype=np.float32)
    
    for (x, y) in points:
        if 0 <= x < w and 0 <= y < h:
            density[int(y), int(x)] = 1
    
    density = gaussian_filter(density, sigma=sigma, truncate=4.0)
    return density

# Process ShanghaiTech Part B
def process_shanghaitech_partB():
    root = 'C:/Users/Dell/Desktop/NN&DL lab/NN-DL-Lab/ShanghaiTech/part_B'
    save_path = 'C:/Users/Dell/Desktop/NN&DL lab/NN-DL-Lab/processed/part_B/'
    os.makedirs(save_path + 'images', exist_ok=True)
    os.makedirs(save_path + 'density', exist_ok=True)

    for split in ['train_data', 'test_data']:
        img_dir = os.path.join(root, split, 'images')
        gt_dir = os.path.join(root, split, 'ground_truth')
        
        for img_file in os.listdir(img_dir):
            if not img_file.endswith('.jpg'): continue
            img_path = os.path.join(img_dir, img_file)
            mat_path = os.path.join(gt_dir, 'GT_' + img_file.replace('.jpg', '.mat'))
            
            mat = sio.loadmat(mat_path)
            points = mat['image_info'][0][0][0][0][0].astype(int)
            
            density = generate_density_map(img_path, points)
            
            # Save
            cv2.imwrite(os.path.join(save_path, 'images', img_file), cv2.imread(img_path))
            np.save(os.path.join(save_path, 'density', img_file.replace('.jpg', '.npy')), density)

process_shanghaitech_partB()