import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import gc
from tqdm.auto import tqdm
import wandb # Import W&B

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Ensure Kaggle paths are correct
# Use this for Kaggle:
DATA_DIR = "/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2"
# Use this for local testing if your "v2" folder is in the same directory:
# DATA_DIR = "./v2/" 

OUTPUT_DIR = "/kaggle/working/output_images/"
CHECKPOINT_DIR = "/kaggle/working/checkpoints/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Hyperparameters
LEARNING_RATE_GEN = 2e-4
LEARNING_RATE_DISC = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = 100 # Adjust for full training, 5-10 for quick test
IMG_SIZE = 256
CHANNELS_IMG_GRAY = 1
CHANNELS_IMG_COLOR = 3
LAMBDA_CYCLE = 10
LAMBDA_L1 = 100
LAMBDA_CONTENT = 1
VGG_CONTENT_LAYERS_INDICES = {'block3_conv3': 16, 'block4_conv3': 23}
CHOSEN_VGG_LAYER_INDEX = VGG_CONTENT_LAYERS_INDICES['block3_conv3']

# Seed for reproducibility
MANUAL_SEED = 42
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(MANUAL_SEED)

# # --- WandB Setup ---
# try:
#     from kaggle_secrets import UserSecretsClient
#     user_secrets = UserSecretsClient()
#     wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
#     wandb.login(key=wandb_api_key)
#     anonymous_wandb = None
# except:
#     anonymous_wandb = "must" # Fallback if secrets not available or not on Kaggle
#     print("W&B API Key not found in Kaggle Secrets or not running on Kaggle. Logging to W&B anonymously.")

# wandb.init(
#     project="SARcyclegan-colorization-advanced", # Choose your project name
#     entity=None, # Your W&B username or team name (optional if logging personally)
#     anonymous=anonymous_wandb,
#     config={
#         "learning_rate_gen": LEARNING_RATE_GEN,
#         "learning_rate_disc": LEARNING_RATE_DISC,
#         "batch_size": BATCH_SIZE,
#         "num_epochs": NUM_EPOCHS,
#         "img_size": IMG_SIZE,
#         "lambda_cycle": LAMBDA_CYCLE,
#         "lambda_l1": LAMBDA_L1,
#         "lambda_content": LAMBDA_CONTENT,
#         "chosen_vgg_layer_idx": CHOSEN_VGG_LAYER_INDEX,
#         "dataset": "Sentinel12-Terrain",
#         "architecture": "CycleGAN-Pix2Pix-VGG-UNet",
#         "device": DEVICE
#     }
# )
# # Save a copy of the notebook to W&B
# # Note: This might not work perfectly in all Kaggle environments or if notebook is too large.
# # wandb.save('*.ipynb') # Saves the current notebook

# # --- 1. Data Preparation ---
# class SentinelDataset(Dataset):
#     def __init__(self, root_dir, terrain_classes=None, transform=None, img_size=256):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.img_size = img_size
#         self.s1_files = []
#         self.s2_files = []

#         if not os.path.exists(root_dir):
#             print(f"Error: Root directory {root_dir} not found. Please check DATA_DIR.")
#             return

#         if terrain_classes is None:
#             try:
#                 terrain_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
#             except FileNotFoundError:
#                 print(f"Error: Could not list directories in {root_dir}. It might be an invalid path or empty.")
#                 return
        
#         if not terrain_classes:
#             print(f"No terrain subdirectories found in {root_dir}.")


#         for terrain in terrain_classes:
#             s1_path = os.path.join(root_dir, terrain, "s1")
#             s2_path = os.path.join(root_dir, terrain, "s2")
#             if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
#                 print(f"Warning: s1 or s2 path not found for terrain {terrain} under {root_dir}")
#                 continue

#             for s1_img_name in os.listdir(s1_path)[:1000]:
#                 s2_img_name = s1_img_name.replace("_s1_", "_s2_")
#                 s1_full_path = os.path.join(s1_path, s1_img_name)
#                 s2_full_path = os.path.join(s2_path, s2_img_name)
#                 if os.path.exists(s2_full_path):
#                     self.s1_files.append(s1_full_path)
#                     self.s2_files.append(s2_full_path)

#         print(f"Found {len(self.s1_files)} paired images from {len(terrain_classes)} terrain class(es).")
#         if len(self.s1_files) == 0:
#             print("Warning: No image pairs were loaded. Check your DATA_DIR and dataset structure.")
#             print(f"DATA_DIR was: {self.root_dir}")
#             print(f"Terrain classes considered: {terrain_classes}")


#     def __len__(self):
#         return len(self.s1_files)

#     def __getitem__(self, idx):
#         s1_img_path = self.s1_files[idx]
#         s2_img_path = self.s2_files[idx]
#         img_s1 = Image.open(s1_img_path).convert("L")
#         img_s2 = Image.open(s2_img_path).convert("RGB")

#         if self.transform:
#             img_s1 = self.transform[0](img_s1)
#             img_s2 = self.transform[1](img_s2)
#         return img_s1, img_s2

# transform_gray = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])
# transform_color = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # --- Dummy data generation for local testing (if dataset not present on local machine) ---
# def create_dummy_data_if_needed(base_path="./v2/", classes=["agri"], num_images=5):
#     first_class_s1_path = os.path.join(base_path, classes[0], "s1")
#     needs_dummy_data = not os.path.exists(base_path) or \
#                        not os.path.exists(first_class_s1_path) or \
#                        not any(os.scandir(first_class_s1_path))

#     if needs_dummy_data and (base_path.startswith("./") or base_path.startswith("../")): # Only for local relative paths
#         print("Creating dummy data for local testing...")
#         for class_name in classes:
#             for s_type in ["s1", "s2"]:
#                 path = os.path.join(base_path, class_name, s_type)
#                 os.makedirs(path, exist_ok=True)
#                 for i in range(num_images):
#                     s1_name_part = f"ROIs_dummy_s1_{i}.png"
#                     s2_name_part = f"ROIs_dummy_s2_{i}.png"
#                     img_name = s1_name_part if s_type == "s1" else s2_name_part

#                     try:
#                         channels = 1 if s_type == "s1" else 3
#                         dummy_array = np.random.randint(0, 256, size=(64, 64, channels), dtype=np.uint8)
#                         if channels == 1:
#                             dummy_array = dummy_array.squeeze(-1)
#                             img = Image.fromarray(dummy_array, mode='L')
#                         else:
#                             img = Image.fromarray(dummy_array, mode='RGB')
#                         img.save(os.path.join(path, img_name))
#                     except Exception as e:
#                         print(f"Error creating dummy image {img_name}: {e}")
#         print("Dummy data created.")

# if DATA_DIR.startswith("./") or DATA_DIR.startswith("../"): # Likely local run
#      create_dummy_data_if_needed(DATA_DIR)

# --- 3. Core Model Architectures ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, features, bn=False, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = self.conv_block(features, features * 2, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = self.conv_block(features * 2, features * 4, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = self.conv_block(features * 4, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc5 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc6 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.enc7 = self.conv_block(features * 8, features * 8, act_fn=nn.LeakyReLU(0.2, inplace=True))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = self.deconv_block(features * 8, features * 8, drop=True)
        self.dec2 = self.deconv_block(features * 8 * 2, features * 8, drop=True)
        self.dec3 = self.deconv_block(features * 8 * 2, features * 8, drop=True)
        self.dec4 = self.deconv_block(features * 8 * 2, features * 8)
        self.dec5 = self.deconv_block(features * 8 * 2, features * 4)
        self.dec6 = self.deconv_block(features * 4 * 2, features * 2)
        self.dec7 = self.deconv_block(features * 2 * 2, features)
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1), nn.Tanh()
        )

    def conv_block(self, ic, oc, bn=True, act_fn=nn.ReLU(True), ks=4, s=2, p=1):
        layers = [nn.Conv2d(ic, oc, ks, s, p, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(oc))
        layers.append(act_fn)
        return nn.Sequential(*layers)

    def deconv_block(self, ic, oc, bn=True, drop=False, act_fn=nn.ReLU(True), ks=4, s=2, p=1):
        layers = [nn.ConvTranspose2d(ic, oc, ks, s, p, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(oc))
        if drop: layers.append(nn.Dropout(0.5))
        layers.append(act_fn)
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        e5 = self.enc5(e4); e6 = self.enc6(e5); e7 = self.enc7(e6)
        b = self.bottleneck(e7)
        d1 = torch.cat([self.dec1(b), e7], 1); d2 = torch.cat([self.dec2(d1), e6], 1)
        d3 = torch.cat([self.dec3(d2), e5], 1); d4 = torch.cat([self.dec4(d3), e4], 1)
        d5 = torch.cat([self.dec5(d4), e3], 1); d6 = torch.cat([self.dec6(d5), e2], 1)
        d7 = torch.cat([self.dec7(d6), e1], 1)
        return self.final_conv(d7)

class Discriminator(nn.Module):
    def __init__(self, in_channels_x=1, in_channels_y=3, features=64):
        super().__init__()
        total_in_channels = in_channels_x + in_channels_y
        self.model = nn.Sequential(
            nn.Conv2d(total_in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            self.disc_block(features, features*2),
            self.disc_block(features*2, features*4),
            self.disc_block(features*4, features*8, s=1), # Stride 1 for last conv before output
            nn.Conv2d(features*8, 1, 4, 1, 1, bias=False) # Output layer
        )
    def disc_block(self, ic, oc, bn=True, ks=4, s=2, p=1):
        layers = [nn.Conv2d(ic, oc, ks, s, p, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(oc))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)

class VGGContentLoss(nn.Module):
    def __init__(self, feature_layer_index=CHOSEN_VGG_LAYER_INDEX, device=DEVICE):
        super().__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:feature_layer_index + 1].to(device)
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for param in self.vgg.parameters(): param.requires_grad = False
    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        x_transformed = (x + 1) / 2.0
        x_transformed = self.transform(x_transformed)
        return self.vgg(x_transformed)

# # --- Model Initialization ---
# gen_AB = UNetGenerator(in_channels=CHANNELS_IMG_GRAY, out_channels=CHANNELS_IMG_COLOR).to(DEVICE)
# gen_BA = UNetGenerator(in_channels=CHANNELS_IMG_COLOR, out_channels=CHANNELS_IMG_GRAY).to(DEVICE)
# disc_Y = Discriminator(in_channels_x=CHANNELS_IMG_GRAY, in_channels_y=CHANNELS_IMG_COLOR).to(DEVICE)
# disc_X = Discriminator(in_channels_x=CHANNELS_IMG_COLOR, in_channels_y=CHANNELS_IMG_GRAY).to(DEVICE)

# gen_AB.apply(weights_init); gen_BA.apply(weights_init)
# disc_X.apply(weights_init); disc_Y.apply(weights_init)

# vgg_content_loss_net = VGGContentLoss().to(DEVICE); vgg_content_loss_net.eval()

# # --- Optimizers ---
# opt_gen = optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999))
# opt_disc = optim.Adam(list(disc_X.parameters()) + list(disc_Y.parameters()), lr=LEARNING_RATE_DISC, betas=(0.5, 0.999))

# # --- Loss Functions ---
# bce_loss = nn.BCEWithLogitsLoss()
# l1_loss = nn.L1Loss()

# # --- Data Loading ---
# terrain_classes_to_use = ['agri']
# dataset = SentinelDataset(root_dir=DATA_DIR, terrain_classes=terrain_classes_to_use,
#                           transform=(transform_gray, transform_color), img_size=IMG_SIZE)
# if len(dataset) == 0:
#     print("CRITICAL: Dataset is empty. Halting execution.")
#     wandb.log({"status": "Dataset empty, run halted."})
#     wandb.finish()
#     exit()

# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

# # --- W&B Watch Models (optional, call after model and optimizer definition, before training loop) ---
# # wandb.watch((gen_AB, gen_BA, disc_X, disc_Y), log="all", log_freq=max(100, len(dataloader)//2), log_graph=False)

# # --- Utility function to save some example images ---
# def save_some_examples(gen_AB_model, val_dataloader, current_epoch, output_folder, current_device, log_to_wandb=True):
#     gen_AB_model.eval()
#     with torch.no_grad():
#         try:
#             val_iter = iter(val_dataloader)
#             x_gray, y_color = next(val_iter)
#         except StopIteration:
#             print("Validation dataloader exhausted or empty for saving examples.")
#             gen_AB_model.train()
#             return
#         except Exception as e:
#             print(f"Error getting validation batch for saving examples: {e}")
#             gen_AB_model.train()
#             return

#         x_gray, y_color = x_gray.to(current_device), y_color.to(current_device)
#         fake_color = gen_AB_model(x_gray)

#         # Denormalize for saving/plotting
#         x_gray_save = (x_gray * 0.5 + 0.5)
#         y_color_save = (y_color * 0.5 + 0.5)
#         fake_color_save = (fake_color * 0.5 + 0.5)

#         # Save locally
#         if BATCH_SIZE == 1: # Simplify for batch size 1
#             x_g_disp = x_gray_save.squeeze(0).squeeze(0).cpu().numpy() # (H,W)
#             y_c_disp = y_color_save.squeeze(0).permute(1,2,0).cpu().numpy() # (H,W,C)
#             f_c_disp = fake_color_save.squeeze(0).permute(1,2,0).cpu().numpy() # (H,W,C)

#             fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#             axs[0].imshow(x_g_disp, cmap='gray'); axs[0].set_title("Input S1"); axs[0].axis("off")
#             axs[1].imshow(y_c_disp); axs[1].set_title("Target S2"); axs[1].axis("off")
#             axs[2].imshow(f_c_disp); axs[2].set_title(f"Gen S2 (Ep {current_epoch+1})"); axs[2].axis("off")
#             plt.tight_layout()
#             save_path = os.path.join(output_folder, f"epoch_{current_epoch+1}_sample.png")
#             plt.savefig(save_path); plt.close(fig)

#             if log_to_wandb:
#                 wandb.log({
#                     "Examples/Input_Grayscale_S1": wandb.Image(x_gray_save.squeeze(0), caption=f"Epoch {current_epoch+1} Input S1"),
#                     "Examples/Target_Color_S2": wandb.Image(y_color_save.squeeze(0), caption=f"Epoch {current_epoch+1} Target S2"),
#                     "Examples/Generated_Color": wandb.Image(fake_color_save.squeeze(0), caption=f"Epoch {current_epoch+1} Generated"),
#                     "epoch": current_epoch + 1 # Ensure images are logged against the correct epoch
#                 })
#         else: # Handle larger batch sizes if you want to log multiple images
#             print("Image saving/logging for batch_size > 1 not fully implemented for individual display.")

#     gen_AB_model.train()

# # --- Training Loop ---
# g_losses_epoch_avg = []
# d_losses_epoch_avg = []

# for epoch in range(NUM_EPOCHS):
#     loop = tqdm(dataloader, leave=True, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
#     current_G_loss_epoch_sum = 0
#     current_D_loss_epoch_sum = 0
#     num_batches_processed = 0

#     for batch_idx, (real_A_gray, real_B_color) in enumerate(loop):
#         real_A_gray = real_A_gray.to(DEVICE)
#         real_B_color = real_B_color.to(DEVICE)

#         # --- Train Discriminators ---
#         for param in gen_AB.parameters(): param.requires_grad = False
#         for param in gen_BA.parameters(): param.requires_grad = False
#         for param in disc_X.parameters(): param.requires_grad = True
#         for param in disc_Y.parameters(): param.requires_grad = True
#         opt_disc.zero_grad(set_to_none=True)

#         fake_B_color = gen_AB(real_A_gray)
#         D_Y_real_pred = disc_Y(real_A_gray, real_B_color)
#         loss_D_Y_real = bce_loss(D_Y_real_pred, torch.ones_like(D_Y_real_pred))
#         D_Y_fake_pred = disc_Y(real_A_gray, fake_B_color.detach())
#         loss_D_Y_fake = bce_loss(D_Y_fake_pred, torch.zeros_like(D_Y_fake_pred))
#         loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5
#         loss_D_Y.backward()

#         fake_A_gray = gen_BA(real_B_color)
#         D_X_real_pred = disc_X(real_B_color, real_A_gray)
#         loss_D_X_real = bce_loss(D_X_real_pred, torch.ones_like(D_X_real_pred))
#         D_X_fake_pred = disc_X(real_B_color, fake_A_gray.detach())
#         loss_D_X_fake = bce_loss(D_X_fake_pred, torch.zeros_like(D_X_fake_pred))
#         loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5
#         loss_D_X.backward()
#         opt_disc.step()
        
#         total_loss_D_batch = loss_D_X + loss_D_Y
#         current_D_loss_epoch_sum += total_loss_D_batch.item()

#         # --- Train Generators ---
#         for param in disc_X.parameters(): param.requires_grad = False
#         for param in disc_Y.parameters(): param.requires_grad = False
#         for param in gen_AB.parameters(): param.requires_grad = True
#         for param in gen_BA.parameters(): param.requires_grad = True
#         opt_gen.zero_grad(set_to_none=True)

#         # Re-use fake_B_color and fake_A_gray if already computed and grads for G are needed
#         # If they were detached for D training, recompute or ensure they are part of G's graph
#         # Here, fake_B_color and fake_A_gray are already computed with G's graph active
#         # (though fake_B_color was used by D, its graph for G is still intact)
#         # No, fake_B_color & fake_A_gray were from previous step, let's ensure they are current
#         fake_B_color_for_G = gen_AB(real_A_gray) # For G_AB losses
#         fake_A_gray_for_G = gen_BA(real_B_color) # For G_BA losses

#         # Adv Loss G_AB
#         D_Y_fake_pred_for_G = disc_Y(real_A_gray, fake_B_color_for_G)
#         loss_adv_G_AB = bce_loss(D_Y_fake_pred_for_G, torch.ones_like(D_Y_fake_pred_for_G))
#         # Adv Loss G_BA
#         D_X_fake_pred_for_F = disc_X(real_B_color, fake_A_gray_for_G)
#         loss_adv_G_BA = bce_loss(D_X_fake_pred_for_F, torch.ones_like(D_X_fake_pred_for_F))
#         # Cycle Loss A
#         cycled_A_gray = gen_BA(fake_B_color_for_G)
#         loss_cycle_A = l1_loss(cycled_A_gray, real_A_gray) * LAMBDA_CYCLE
#         # Cycle Loss B
#         cycled_B_color = gen_AB(fake_A_gray_for_G)
#         loss_cycle_B = l1_loss(cycled_B_color, real_B_color) * LAMBDA_CYCLE
#         # L1 Loss G_AB
#         loss_L1_G_AB = l1_loss(fake_B_color_for_G, real_B_color) * LAMBDA_L1
#         # VGG Content Loss G_AB
#         with torch.no_grad(): real_B_features = vgg_content_loss_net(real_B_color)
#         fake_B_features = vgg_content_loss_net(fake_B_color_for_G)
#         loss_content_G_AB = l1_loss(fake_B_features, real_B_features.detach()) * LAMBDA_CONTENT
        
#         loss_G_batch = loss_adv_G_AB + loss_adv_G_BA + loss_cycle_A + loss_cycle_B + loss_L1_G_AB + loss_content_G_AB
#         loss_G_batch.backward()
#         opt_gen.step()
#         current_G_loss_epoch_sum += loss_G_batch.item()
        
#         num_batches_processed += 1

#         # Log batch metrics to W&B
#         log_metrics_batch = {
#             "Batch/D_Loss_X": loss_D_X.item(), "Batch/D_Loss_Y": loss_D_Y.item(),
#             "Batch/Total_D_Loss": total_loss_D_batch.item(),
#             "Batch/G_Loss_Adv_AB": loss_adv_G_AB.item(), "Batch/G_Loss_Adv_BA": loss_adv_G_BA.item(),
#             "Batch/G_Loss_L1_AB": loss_L1_G_AB.item(), "Batch/G_Loss_Content_AB": loss_content_G_AB.item(),
#             "Batch/G_Loss_Cycle_A": loss_cycle_A.item(), "Batch/G_Loss_Cycle_B": loss_cycle_B.item(),
#             "Batch/Total_G_Loss": loss_G_batch.item(),
#             "epoch": epoch + 1 # Log epoch for grouping
#         }
#         wandb.log(log_metrics_batch)
        
#         loop.set_postfix({k.split('/')[-1]: f"{v:.3f}" for k,v in log_metrics_batch.items() if isinstance(v,float)})
        
#         del real_A_gray, real_B_color, fake_A_gray, fake_B_color, cycled_A_gray, cycled_B_color
#         del fake_A_gray_for_G, fake_B_color_for_G
#         del D_Y_real_pred, D_Y_fake_pred, D_X_real_pred, D_X_fake_pred
#         del D_Y_fake_pred_for_G, D_X_fake_pred_for_F
#         if DEVICE == "cuda": torch.cuda.empty_cache()

#     # End of Epoch
#     avg_G_loss_this_epoch = current_G_loss_epoch_sum / num_batches_processed if num_batches_processed > 0 else 0
#     avg_D_loss_this_epoch = current_D_loss_epoch_sum / num_batches_processed if num_batches_processed > 0 else 0
#     g_losses_epoch_avg.append(avg_G_loss_this_epoch)
#     d_losses_epoch_avg.append(avg_D_loss_this_epoch)

#     # Log epoch-average metrics to W&B
#     wandb.log({
#         "Epoch/Avg_G_Loss": avg_G_loss_this_epoch,
#         "Epoch/Avg_D_Loss": avg_D_loss_this_epoch,
#         "epoch": epoch + 1 # Crucial for x-axis in W&B if you want epoch number
#     })
#     print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Avg G Loss: {avg_G_loss_this_epoch:.4f}, Avg D Loss: {avg_D_loss_this_epoch:.4f}")

#     if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1: # Save every 5 epochs and last epoch
#         save_some_examples(gen_AB, dataloader, epoch, output_folder=OUTPUT_DIR, current_device=DEVICE, log_to_wandb=True)
        
#         # Save checkpoints locally
#         gen_ab_path = os.path.join(CHECKPOINT_DIR, f"gen_AB_epoch_{epoch+1}.pth")
#         # ... (save all models)
#         torch.save(gen_AB.state_dict(), gen_ab_path)
#         torch.save(gen_BA.state_dict(), os.path.join(CHECKPOINT_DIR, f"gen_BA_epoch_{epoch+1}.pth"))
#         torch.save(disc_X.state_dict(), os.path.join(CHECKPOINT_DIR, f"disc_X_epoch_{epoch+1}.pth"))
#         torch.save(disc_Y.state_dict(), os.path.join(CHECKPOINT_DIR, f"disc_Y_epoch_{epoch+1}.pth"))
#         print(f"Local checkpoints saved for epoch {epoch+1}")

#         # Log model artifact to W&B
#         try:
#             model_artifact_name = f'{wandb.run.name}-epoch-{epoch+1}' if wandb.run.name else f'model-epoch-{epoch+1}'
#             artifact = wandb.Artifact(model_artifact_name, type='model',
#                                       description=f"CycleGAN models after epoch {epoch+1}",
#                                       metadata=wandb.config.as_dict()) # Add config to artifact metadata
#             artifact.add_file(gen_ab_path, name="gen_AB.pth")
#             artifact.add_file(os.path.join(CHECKPOINT_DIR, f"gen_BA_epoch_{epoch+1}.pth"), name="gen_BA.pth")
#             # Add other models if needed
#             wandb.log_artifact(artifact)
#             print(f"Model artifact logged to W&B for epoch {epoch+1}")
#         except Exception as e:
#             print(f"Error logging artifact to W&B: {e}")


#     gc.collect(); 
#     if DEVICE == "cuda": torch.cuda.empty_cache()

# # --- Plot and save final losses locally ---
# plt.figure(figsize=(10, 5))
# plt.title("Generator and Discriminator Avg Loss During Training (Local Plot)")
# plt.plot(g_losses_epoch_avg, label="Avg G Loss")
# plt.plot(d_losses_epoch_avg, label="Avg D Loss")
# plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()
# local_loss_plot_path = os.path.join(OUTPUT_DIR, "loss_plot_avg_epoch.png")
# plt.savefig(local_loss_plot_path)
# # plt.show() # Might not work well in pure script mode on Kaggle, W&B plots are better

# # Log final loss plot to W&B
# wandb.log({"Training_Summary/Avg_Loss_Plot": wandb.Image(local_loss_plot_path)})

# print("Training finished.")
# wandb.finish()
