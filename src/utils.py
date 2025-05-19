# src/utils.py
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True # May impact performance
        torch.backends.cudnn.benchmark = False   # Ensure reproducibility

def save_some_examples(gen_AB_model, val_dataloader, current_epoch, output_folder_local, current_device,
                       mlflow_client=None, batch_size_display=1):
    gen_AB_model.eval()
    with torch.no_grad():
        try:
            val_iter = iter(val_dataloader)
            x_gray, y_color = next(val_iter)
        except StopIteration:
            print("Validation dataloader exhausted for saving examples.")
            gen_AB_model.train()
            return
        except Exception as e:
            print(f"Error getting validation batch for saving examples: {e}")
            gen_AB_model.train()
            return

        x_gray, y_color = x_gray.to(current_device), y_color.to(current_device)
        fake_color = gen_AB_model(x_gray)

        # Denormalize from [-1, 1] to [0, 1] for saving/plotting
        x_gray_save = (x_gray * 0.5 + 0.5).clamp(0, 1)
        y_color_save = (y_color * 0.5 + 0.5).clamp(0, 1)
        fake_color_save = (fake_color * 0.5 + 0.5).clamp(0, 1)

        os.makedirs(output_folder_local, exist_ok=True)
        
        num_to_display = min(x_gray_save.size(0), batch_size_display)

        for i in range(num_to_display):
            # Squeeze batch dim for single image, then channel dim for grayscale
            x_g_disp = x_gray_save[i].squeeze(0).cpu().numpy() 
            y_c_disp = y_color_save[i].permute(1,2,0).cpu().numpy() 
            f_c_disp = fake_color_save[i].permute(1,2,0).cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(x_g_disp, cmap='gray'); axs[0].set_title("Input S1 (Source)"); axs[0].axis("off")
            axs[1].imshow(y_c_disp); axs[1].set_title("Target S2 (Real)"); axs[1].axis("off")
            axs[2].imshow(f_c_disp); axs[2].set_title(f"Generated S2 (Epoch {current_epoch+1})"); axs[2].axis("off")
            plt.tight_layout()
            
            local_image_save_path = os.path.join(output_folder_local, f"epoch_{current_epoch+1}_sample_{i}.png")
            plt.savefig(local_image_save_path)
            plt.close(fig)
            print(f"Saved example image to {local_image_save_path}")

            if mlflow_client: 
                try:
                    mlflow_artifact_path_dir = f"epoch_examples/epoch_{current_epoch+1}"
                    mlflow_client.log_artifact(local_image_save_path, artifact_path=mlflow_artifact_path_dir)
                    print(f"Logged {local_image_save_path} to MLflow artifacts: {mlflow_artifact_path_dir}")
                except Exception as e:
                    print(f"Error logging example image to MLflow: {e}")
    gen_AB_model.train()
