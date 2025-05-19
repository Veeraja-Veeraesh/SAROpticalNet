# src/data_loader.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import io 

from google.cloud import storage
from google.api_core.exceptions import NotFound

# --- Worker-local GCS client initialization ---
_GCS_CLIENT_WORKER_CACHE = {} # Cache client per worker PID
_GCS_BUCKET_WORKER_CACHE = {} # Cache bucket per (PID, bucket_name)

def _get_worker_gcs_client(gcp_project_id_for_worker):
    pid = os.getpid()
    if pid not in _GCS_CLIENT_WORKER_CACHE:
        if not gcp_project_id_for_worker:
            # This should ideally not happen if configured correctly, but as a fallback:
            print(f"Warning (Worker {pid}): gcp_project_id_for_worker not explicitly passed to _get_worker_gcs_client. Attempting client init without project.")
            _GCS_CLIENT_WORKER_CACHE[pid] = storage.Client()
        else:
            _GCS_CLIENT_WORKER_CACHE[pid] = storage.Client(project=gcp_project_id_for_worker)
        print(f"Worker {pid} initialized GCS client (Project: {gcp_project_id_for_worker if gcp_project_id_for_worker else 'Default'}).")
    return _GCS_CLIENT_WORKER_CACHE[pid]

def _get_worker_gcs_bucket(bucket_name, gcp_project_id_for_worker):
    pid = os.getpid()
    cache_key = (pid, bucket_name)
    if cache_key not in _GCS_BUCKET_WORKER_CACHE:
        client = _get_worker_gcs_client(gcp_project_id_for_worker)
        _GCS_BUCKET_WORKER_CACHE[cache_key] = client.bucket(bucket_name)
        print(f"Worker {pid} accessed GCS bucket: {bucket_name}")
    return _GCS_BUCKET_WORKER_CACHE[cache_key]
# --- End Worker-local GCS client ---


class SentinelDataset(Dataset):
    def __init__(self, root_dir, terrain_classes=None, transform_gray=None, transform_color=None,
                 img_size=256, max_images_per_class=1000, gcs_bucket_name=None, gcp_project_id=None):
        self.root_dir = root_dir
        self.transform_gray = transform_gray
        self.transform_color = transform_color
        self.img_size = img_size
        self.s1_files = [] # Stores full GCS blob names or local paths
        self.s2_files = []
        self.is_gcs = root_dir.startswith("gs://")
        
        # Store these for worker processes to re-initialize their own clients
        self.gcs_bucket_name_for_worker = None
        self.gcs_prefix_for_worker = ""
        self.gcp_project_id_for_worker = gcp_project_id # Store project ID for workers

        # This client is temporary, used only in __init__ for listing, not stored on self.
        # It must be initialized with the project ID.
        init_gcs_client = None
        init_gcs_bucket = None

        if self.is_gcs:
            if not gcp_project_id:
                raise ValueError("gcp_project_id is required for GCS operations but was not provided to SentinelDataset.")
            
            # Determine bucket name and prefix
            if not gcs_bucket_name:
                try:
                    self.gcs_bucket_name_for_worker = root_dir.split("gs://")[1].split("/")[0]
                except IndexError:
                    raise ValueError("GCS path provided for root_dir, but gcs_bucket_name could not be parsed and was not explicitly provided.")
            else:
                self.gcs_bucket_name_for_worker = gcs_bucket_name
            
            self.gcs_prefix_for_worker = "/".join(root_dir.split("gs://")[1].split("/")[1:])
            if self.gcs_prefix_for_worker and not self.gcs_prefix_for_worker.endswith('/'):
                self.gcs_prefix_for_worker += '/'

            print(f"Initializing GCS client in __init__ (main process) for listing. Project: {gcp_project_id}, Bucket: {self.gcs_bucket_name_for_worker}")
            init_gcs_client = storage.Client(project=gcp_project_id) # Correctly pass project_id
            init_gcs_bucket = init_gcs_client.bucket(self.gcs_bucket_name_for_worker)

        else: # Local filesystem
            if not os.path.exists(root_dir):
                print(f"Error: Local root directory {root_dir} not found.")
                # Consider raising an error or returning if this is critical
                return 

        # --- File Listing Logic (uses init_gcs_bucket if GCS, or os.listdir if local) ---
        if terrain_classes is None or not terrain_classes:
            print(f"Attempting to list all terrain classes from {root_dir}...")
            if self.is_gcs:
                if not init_gcs_bucket: # Should not happen if self.is_gcs is true and previous checks pass
                    raise RuntimeError("GCS setup error: init_gcs_bucket not initialized for listing.")
                iterator = init_gcs_bucket.list_blobs(prefix=self.gcs_prefix_for_worker, delimiter='/')
                terrain_classes = [p.replace(self.gcs_prefix_for_worker, "").strip('/') for p in iterator.prefixes]
            else: # Local
                try:
                    terrain_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
                except FileNotFoundError:
                    print(f"Error: Could not list directories in {root_dir}.")
                    return # Or raise
        
        if not terrain_classes:
            print(f"No terrain subdirectories found or specified in {root_dir}.")
            return
        
        print(f"Processing terrain classes: {terrain_classes}")

        for terrain in terrain_classes:
            s1_path_suffix = os.path.join(terrain, "s1").replace("\\", "/") + "/" 
            s2_path_suffix = os.path.join(terrain, "s2").replace("\\", "/") + "/"

            current_terrain_s1_files = []
            current_terrain_s2_files = []

            if self.is_gcs:
                if not init_gcs_bucket: raise RuntimeError("GCS init error.") # Safeguard
                s1_full_gcs_prefix = self.gcs_prefix_for_worker + s1_path_suffix
                # List blobs directly from init_gcs_bucket
                s1_blobs_iterator = init_gcs_bucket.list_blobs(prefix=s1_full_gcs_prefix)
                s1_image_names_gcs = [blob.name for blob in s1_blobs_iterator if not blob.name.endswith('/') and blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                
                for s1_blob_full_name in s1_image_names_gcs:
                    s1_img_name_only = os.path.basename(s1_blob_full_name)
                    s2_img_name_only = s1_img_name_only.replace("_s1_", "_s2_") 
                    s2_blob_full_name = self.gcs_prefix_for_worker + s2_path_suffix + s2_img_name_only
                    
                    s2_blob = init_gcs_bucket.blob(s2_blob_full_name)
                    if s2_blob.exists():
                        current_terrain_s1_files.append(s1_blob_full_name)
                        current_terrain_s2_files.append(s2_blob_full_name)
            else: # Local filesystem logic (remains the same)
                s1_local_path = os.path.join(root_dir, terrain, "s1")
                s2_local_path = os.path.join(root_dir, terrain, "s2")
                if not (os.path.exists(s1_local_path) and os.path.exists(s2_local_path)):
                    print(f"Warning: Local s1 or s2 path not found for terrain {terrain} under {root_dir}")
                    continue
                for s1_img_name_local in os.listdir(s1_local_path):
                    if not s1_img_name_local.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        continue
                    s2_img_name_local = s1_img_name_local.replace("_s1_", "_s2_")
                    s1_full_local_path = os.path.join(s1_local_path, s1_img_name_local)
                    s2_full_local_path = os.path.join(s2_local_path, s2_img_name_local)
                    if os.path.exists(s2_full_local_path):
                        current_terrain_s1_files.append(s1_full_local_path)
                        current_terrain_s2_files.append(s2_full_local_path)

            limit = min(len(current_terrain_s1_files), max_images_per_class)
            self.s1_files.extend(current_terrain_s1_files[:limit])
            self.s2_files.extend(current_terrain_s2_files[:limit])

        print(f"Found {len(self.s1_files)} paired images from terrain class(es): {terrain_classes} in {'GCS' if self.is_gcs else 'local FS'}.")
        if len(self.s1_files) == 0:
            print(f"Warning: No image pairs were loaded. Check data source ('{self.root_dir}') and structure for classes: {terrain_classes}.")
        # init_gcs_client and init_gcs_bucket go out of scope here and are not stored on self

    def __len__(self):
        return len(self.s1_files)

    def _load_image_from_path(self, path_or_blob_name, mode):
        try:
            if self.is_gcs:
                # Use worker-local GCS client and bucket
                # Pass the stored gcp_project_id_for_worker
                bucket = _get_worker_gcs_bucket(self.gcs_bucket_name_for_worker, self.gcp_project_id_for_worker)
                blob = bucket.blob(path_or_blob_name)
                img_bytes = blob.download_as_bytes()
                img = Image.open(io.BytesIO(img_bytes)).convert(mode)
            else: # Local
                img = Image.open(path_or_blob_name).convert(mode)
            return img
        except (NotFound, FileNotFoundError):
            print(f"Warning (Worker {os.getpid()}): Image not found - {path_or_blob_name}. Returning dummy image.")
        except Exception as e:
            print(f"Warning (Worker {os.getpid()}): Error loading image {path_or_blob_name}: {e}. Returning dummy image.")
        
        return Image.new(mode, (self.img_size, self.img_size), color='gray' if mode == 'L' else 'black')

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

# get_transforms function remains the same
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
