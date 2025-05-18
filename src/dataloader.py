# src/data_loader.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import io # For handling bytes from GCS

# GCP Specific
from google.cloud import storage
from google.api_core.exceptions import NotFound

# Default IMG_SIZE if not passed (will be overridden by config typically)
# IMG_SIZE_DEFAULT = 256

class SentinelDataset(Dataset):
    def __init__(self, root_dir, terrain_classes=None, transform_gray=None, transform_color=None,
                 img_size=256, max_images_per_class=1000, gcs_bucket_name=None):
        self.root_dir = root_dir
        self.transform_gray = transform_gray
        self.transform_color = transform_color
        self.img_size = img_size
        self.s1_files = []  # Will store GCS paths or local temp paths
        self.s2_files = []
        self.is_gcs = root_dir.startswith("gs://")
        self.gcs_client = None
        self.gcs_bucket = None

        if self.is_gcs:
            if not gcs_bucket_name:
                # Try to parse from root_dir
                try:
                    gcs_bucket_name = root_dir.split("gs://")[1].split("/")[0]
                except IndexError:
                    raise ValueError("GCS path provided but bucket name could not be parsed and gcs_bucket_name not provided.")
            
            print(f"Initializing GCS client for bucket: {gcs_bucket_name}")
            self.gcs_client = storage.Client() # Assumes GOOGLE_APPLICATION_CREDENTIALS is set or gcloud auth
            self.gcs_bucket = self.gcs_client.bucket(gcs_bucket_name)
            # The rest of root_dir is the prefix within the bucket
            self.gcs_prefix = "/".join(root_dir.split("gs://")[1].split("/")[1:])
            if not self.gcs_prefix.endswith('/'):
                self.gcs_prefix += '/'
        else: # Local filesystem
            if not os.path.exists(root_dir):
                print(f"Error: Local root directory {root_dir} not found.")
                return

        if terrain_classes is None:
            if self.is_gcs:
                # Listing "directories" in GCS requires listing blobs with prefixes
                blobs = self.gcs_bucket.list_blobs(prefix=self.gcs_prefix, delimiter='/')
                # Consume the iterator to get prefixes
                list(blobs) # This is just to make blobs.prefixes available if API behaves that way
                terrain_classes = [prefix.replace(self.gcs_prefix, "").strip('/') for prefix in blobs.prefixes]

            else: # Local
                try:
                    terrain_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
                except FileNotFoundError:
                    print(f"Error: Could not list directories in {root_dir}.")
                    return
        
        if not terrain_classes:
            print(f"No terrain subdirectories found in {root_dir}.")
            return
        
        print(f"Processing terrain classes: {terrain_classes}")

        for terrain in terrain_classes:
            s1_path_suffix = os.path.join(terrain, "s1").replace("\\", "/") + "/" # Ensure forward slashes for GCS
            s2_path_suffix = os.path.join(terrain, "s2").replace("\\", "/") + "/"

            current_terrain_s1_files = []
            current_terrain_s2_files = []

            if self.is_gcs:
                s1_full_prefix = self.gcs_prefix + s1_path_suffix
                s1_blobs = list(self.gcs_bucket.list_blobs(prefix=s1_full_prefix))
                s1_image_names_gcs = [blob.name for blob in s1_blobs if not blob.name.endswith('/')] # Exclude "folder" blobs
                
                for s1_blob_full_name in s1_image_names_gcs:
                    s1_img_name_only = os.path.basename(s1_blob_full_name)
                    s2_img_name_only = s1_img_name_only.replace("_s1_", "_s2_") # Assuming consistent naming
                    s2_blob_full_name = self.gcs_prefix + s2_path_suffix + s2_img_name_only
                    
                    # Check if corresponding s2 blob exists
                    s2_blob = self.gcs_bucket.blob(s2_blob_full_name)
                    if s2_blob.exists() and s1_img_name_only.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        current_terrain_s1_files.append(s1_blob_full_name) # Store full GCS path
                        current_terrain_s2_files.append(s2_blob_full_name) # Store full GCS path
            else: # Local filesystem
                s1_local_path = os.path.join(root_dir, terrain, "s1")
                s2_local_path = os.path.join(root_dir, terrain, "s2")

                if not (os.path.exists(s1_local_path) and os.path.exists(s2_local_path)):
                    print(f"Warning: Local s1 or s2 path not found for terrain {terrain} under {root_dir}")
                    continue
                
                s1_image_names_local = os.listdir(s1_local_path)
                for s1_img_name_local in s1_image_names_local:
                    s2_img_name_local = s1_img_name_local.replace("_s1_", "_s2_")
                    s1_full_local_path = os.path.join(s1_local_path, s1_img_name_local)
                    s2_full_local_path = os.path.join(s2_local_path, s2_img_name_local)
                    
                    if os.path.exists(s2_full_local_path) and s1_img_name_local.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        current_terrain_s1_files.append(s1_full_local_path)
                        current_terrain_s2_files.append(s2_full_local_path)

            limit = min(len(current_terrain_s1_files), max_images_per_class)
            self.s1_files.extend(current_terrain_s1_files[:limit])
            self.s2_files.extend(current_terrain_s2_files[:limit])

        print(f"Found {len(self.s1_files)} paired images from terrain class(es): {terrain_classes} in {'GCS' if self.is_gcs else 'local FS'}.")
        if len(self.s1_files) == 0:
            print(f"Warning: No image pairs were loaded. Check your data source ('{self.root_dir}') and structure.")

    def __len__(self):
        return len(self.s1_files)

    def _load_image_from_path(self, path_or_blob_name, mode):
        if self.is_gcs:
            blob = self.gcs_bucket.blob(path_or_blob_name)
            try:
                img_bytes = blob.download_as_bytes()
                img = Image.open(io.BytesIO(img_bytes)).convert(mode)
            except NotFound:
                print(f"GCS Error: Blob not found {path_or_blob_name}")
                return Image.new(mode, (self.img_size, self.img_size)) # Return dummy
            except Exception as e:
                print(f"Error loading image {path_or_blob_name} from GCS: {e}")
                return Image.new(mode, (self.img_size, self.img_size)) # Return dummy
        else: # Local
            try:
                img = Image.open(path_or_blob_name).convert(mode)
            except FileNotFoundError:
                print(f"Local FS Error: File not found {path_or_blob_name}")
                return Image.new(mode, (self.img_size, self.img_size))
            except Exception as e:
                print(f"Error loading image {path_or_blob_name} from local FS: {e}")
                return Image.new(mode, (self.img_size, self.img_size))
        return img

    def __getitem__(self, idx):
        s1_file_ref = self.s1_files[idx]
        s2_file_ref = self.s2_files[idx]
        
        img_s1 = self._load_image_from_path(s1_file_ref, "L")
        img_s2 = self._load_image_from_path(s2_file_ref, "RGB")

        if self.transform_gray:
            img_s1 = self.transform_gray(img_s1)
        if self.transform_color:
            img_s2 = self.transform_color(img_s2)
            
        return img_s1, img_s2

# get_transforms and create_dummy_data_if_needed remain the same as before
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

def create_dummy_data_if_needed(base_path="./data/v2_dummy/", classes=["agri"], num_images=5, img_size=256):
    # Make sure the base_path exists before checking if it's empty
    os.makedirs(base_path, exist_ok=True)
    
    first_class_s1_path = os.path.join(base_path, classes[0], "s1")
    # Check if dummy data needs to be created
    # This condition might need refinement: check if the directory is truly empty or just contains folders
    needs_dummy_data = not os.path.exists(first_class_s1_path) or not any(f for f in os.listdir(first_class_s1_path) if os.path.isfile(os.path.join(first_class_s1_path, f)))


    if needs_dummy_data and (base_path.startswith("./") or base_path.startswith("../")): # Only for local relative paths
        print(f"Creating dummy data in {base_path} for local testing...")
        for class_name in classes:
            for s_type in ["s1", "s2"]:
                path = os.path.join(base_path, class_name, s_type)
                os.makedirs(path, exist_ok=True)
                for i in range(num_images):
                    # Naming convention from original notebook for dummy data
                    # Example: ROIs_dummy_s1_0.png, ROIs_dummy_s2_0.png
                    img_name_base = f"ROIs_dummy_img_{i}"
                    final_name = f"{img_name_base.replace('img', s_type)}.png"


                    try:
                        channels = 1 if s_type == "s1" else 3
                        # Use numpy for array creation then convert to PIL
                        dummy_array_np = torch.randint(0, 256, size=(img_size, img_size, channels), dtype=torch.uint8).numpy()
                        if channels == 1:
                            img = Image.fromarray(dummy_array_np.squeeze(-1), mode='L')
                        else:
                            img = Image.fromarray(dummy_array_np, mode='RGB')
                        img.save(os.path.join(path, final_name))
                    except Exception as e:
                        print(f"Error creating dummy image {final_name} in {path}: {e}")
        print("Dummy data created.")
        return True
    elif not needs_dummy_data:
        print(f"Dummy data directory {base_path} already populated or not needed for GCS.")
    return False
