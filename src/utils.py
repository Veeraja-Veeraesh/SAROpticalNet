# src/utils.py
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
# import wandb # We'll handle wandb logging within train.py or via MLflow

# Constants from model.py or config
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_some_examples(gen_AB_model, val_dataloader, current_epoch, output_folder, current_device, log_to_wandb_or_mlflow=None, batch_size_display=1):
    gen_AB_model.eval()
    with torch.no_grad():
        try:
            val_iter = iter(val_dataloader)
            x_gray, y_color = next(val_iter)
        except StopIteration:
            print("Validation dataloader exhausted or empty for saving examples.")
            gen_AB_model.train()
            return
        except Exception as e:
            print(f"Error getting validation batch for saving examples: {e}")
            gen_AB_model.train()
            return

        x_gray, y_color = x_gray.to(current_device), y_color.to(current_device)
        fake_color = gen_AB_model(x_gray)

        # Denormalize for saving/plotting: assumes normalization to [-1, 1]
        x_gray_save = (x_gray * 0.5 + 0.5).clamp(0, 1)
        y_color_save = (y_color * 0.5 + 0.5).clamp(0, 1)
        fake_color_save = (fake_color * 0.5 + 0.5).clamp(0, 1)

        # Save locally
        os.makedirs(output_folder, exist_ok=True)
        
        # Display and save the first image of the batch if batch_size_display=1
        # or loop through batch_size_display images
        num_to_display = min(x_gray_save.size(0), batch_size_display)

        for i in range(num_to_display):
            x_g_disp = x_gray_save[i].squeeze(0).cpu().numpy() # (H,W)
            y_c_disp = y_color_save[i].permute(1,2,0).cpu().numpy() # (H,W,C)
            f_c_disp = fake_color_save[i].permute(1,2,0).cpu().numpy() # (H,W,C)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(x_g_disp, cmap='gray'); axs[0].set_title("Input S1"); axs[0].axis("off")
            axs[1].imshow(y_c_disp); axs[1].set_title("Target S2"); axs[1].axis("off")
            axs[2].imshow(f_c_disp); axs[2].set_title(f"Gen S2 (Ep {current_epoch+1})"); axs[2].axis("off")
            plt.tight_layout()
            
            save_path = os.path.join(output_folder, f"epoch_{current_epoch+1}_sample_{i}.png")
            plt.savefig(save_path)
            plt.close(fig)

            if log_to_wandb_or_mlflow: # This will be an MLflow client or wandb
                # For MLflow, you'd use: mlflow.log_artifact(save_path, artifact_path="examples")
                # For W&B, it's slightly different
                if hasattr(log_to_wandb_or_mlflow, 'log_artifact'): # MLflow like
                    log_to_wandb_or_mlflow.log_artifact(save_path, artifact_path=f"epoch_examples/epoch_{current_epoch+1}")
                elif hasattr(log_to_wandb_or_mlflow, 'Image'): # W&B like
                     log_to_wandb_or_mlflow.log({
                        f"Examples_Epoch_{current_epoch+1}/Input_S1_Sample_{i}": log_to_wandb_or_mlflow.Image(x_gray_save[i], caption=f"Epoch {current_epoch+1} Input S1"),
                        f"Examples_Epoch_{current_epoch+1}/Target_S2_Sample_{i}": log_to_wandb_or_mlflow.Image(y_color_save[i], caption=f"Epoch {current_epoch+1} Target S2"),
                        f"Examples_Epoch_{current_epoch+1}/Generated_S2_Sample_{i}": log_to_wandb_or_mlflow.Image(fake_color_save[i], caption=f"Epoch {current_epoch+1} Generated"),
                        "epoch": current_epoch + 1
                    })


    gen_AB_model.train()

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
