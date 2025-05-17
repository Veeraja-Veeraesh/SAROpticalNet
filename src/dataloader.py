# src/data_loader.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Constants (can be moved to config.yaml later)
# IMG_SIZE = 256 # This will come from config

class SentinelDataset(Dataset):
    def __init__(self, root_dir, terrain_classes=None, transform_gray=None, transform_color=None, img_size=256, max_images_per_class=1000):
        self.root_dir = root_dir
        self.transform_gray = transform_gray
        self.transform_color = transform_color
        self.img_size = img_size
        self.s1_files = []
        self.s2_files = []

        if not os.path.exists(root_dir):
            print(f"Error: Root directory {root_dir} not found.")
            return

        if terrain_classes is None:
            try:
                terrain_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            except FileNotFoundError:
                print(f"Error: Could not list directories in {root_dir}. It might be an invalid path or empty.")
                return
        
        if not terrain_classes:
            print(f"No terrain subdirectories found in {root_dir}.")

        for terrain in terrain_classes:
            s1_path = os.path.join(root_dir, terrain, "s1")
            s2_path = os.path.join(root_dir, terrain, "s2")
            if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
                print(f"Warning: s1 or s2 path not found for terrain {terrain} under {root_dir}")
                continue

            s1_image_names = os.listdir(s1_path)
            current_terrain_s1_files = []
            current_terrain_s2_files = []

            for s1_img_name in s1_image_names:
                s2_img_name = s1_img_name.replace("_s1_", "_s2_") # Assuming consistent naming
                # You might need more robust pairing logic if names aren't perfectly aligned
                # e.g. common prefix extraction.
                
                s1_full_path = os.path.join(s1_path, s1_img_name)
                s2_full_path = os.path.join(s2_path, s2_img_name)
                
                if os.path.exists(s2_full_path) and s1_img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    current_terrain_s1_files.append(s1_full_path)
                    current_terrain_s2_files.append(s2_full_path)
            
            # Limit images per class
            limit = min(len(current_terrain_s1_files), max_images_per_class)
            self.s1_files.extend(current_terrain_s1_files[:limit])
            self.s2_files.extend(current_terrain_s2_files[:limit])


        print(f"Found {len(self.s1_files)} paired images from terrain class(es): {terrain_classes}.")
        if len(self.s1_files) == 0:
            print("Warning: No image pairs were loaded. Check your DATA_DIR and dataset structure.")
            print(f"DATA_DIR was: {self.root_dir}")
            print(f"Terrain classes considered: {terrain_classes}")

    def __len__(self):
        return len(self.s1_files)

    def __getitem__(self, idx):
        s1_img_path = self.s1_files[idx]
        s2_img_path = self.s2_files[idx]
        
        try:
            img_s1 = Image.open(s1_img_path).convert("L")
            img_s2 = Image.open(s2_img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image pair: {s1_img_path}, {s2_img_path}. Error: {e}")
            # Return a placeholder or skip this sample if robust error handling is needed
            # For simplicity, we'll let it crash or you can return None and handle in DataLoader collate_fn
            # Or return a dummy tensor of correct shape
            dummy_s1 = torch.zeros((1, self.img_size, self.img_size))
            dummy_s2 = torch.zeros((3, self.img_size, self.img_size))
            if self.transform_gray: dummy_s1 = self.transform_gray(Image.new('L', (self.img_size, self.img_size)))
            if self.transform_color: dummy_s2 = self.transform_color(Image.new('RGB', (self.img_size, self.img_size)))
            return dummy_s1, dummy_s2


        if self.transform_gray:
            img_s1 = self.transform_gray(img_s1)
        if self.transform_color:
            img_s2 = self.transform_color(img_s2)
            
        return img_s1, img_s2

def get_transforms(img_size):
    transform_gray = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transform_color = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform_gray, transform_color
