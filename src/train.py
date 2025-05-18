# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import gc
import yaml # For loading config
import mlflow # For MLflow
import mlflow.pytorch # For MLflow PyTorch integration

# Import from our modules
from model import UNetGenerator, Discriminator, VGGContentLoss, weights_init
from data_loader import SentinelDataset, get_transforms, create_dummy_data_if_needed
from utils import save_some_examples, set_seed

# --- WandB Setup (Optional, if you want to keep it alongside MLflow) ---
# We will primarily focus on MLflow for model registry.
import wandb
WANDB_ACTIVE = True # Set to True if you want to use W&B

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    set_seed(config['MANUAL_SEED'])

    DEVICE = config['DEVICE'] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- MLflow Setup ---
    mlflow.set_tracking_uri(config['MLFLOW_TRACKING_URI'])
    try:
        experiment_id = mlflow.create_experiment(config['MLFLOW_EXPERIMENT_NAME'])
    except mlflow.exceptions.MlflowException: # If experiment already exists
        experiment_id = mlflow.get_experiment_by_name(config['MLFLOW_EXPERIMENT_NAME']).experiment_id
    
    mlflow.set_experiment(experiment_id=experiment_id)
    
    # Start MLflow Run
    # Nested runs can be useful if you have hyperparameter tuning, but for a single train:
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.log_params(config) # Log all config parameters

        # --- Data Loading ---
        transform_gray, transform_color = get_transforms(config['IMG_SIZE'])
        dataset = SentinelDataset(
            root_dir=config['DATA_DIR'], # This can now be gs:// path
            terrain_classes=config['TERRAIN_CLASSES_TO_USE'],
            transform_gray=transform_gray,
            transform_color=transform_color,
            img_size=config['IMG_SIZE'],
            max_images_per_class=config['MAX_IMAGES_PER_CLASS'],
            gcs_bucket_name=config.get('GCS_BUCKET_NAME') # Pass bucket name if using GCS
        )

        if len(dataset) == 0:
            print("CRITICAL: Dataset is empty. Halting execution.")
            mlflow.log_metric("status", 0) # 0 for failure
            mlflow.set_terminated("FAILED", "Dataset empty")
            return
        
        print(f"Successfully initialized dataset. Number of items: {len(dataset)}")
        if len(dataset) > 0:
            print("Attempting to load first item...")
            try:
                s1_sample, s2_sample = dataset[0]
                print(f"First S1 sample shape: {s1_sample.shape}, dtype: {s1_sample.dtype}")
                print(f"First S2 sample shape: {s2_sample.shape}, dtype: {s2_sample.dtype}")
            except Exception as e:
                print(f"Error loading first sample: {e}")
        # return # Add this to exit after testing data loading

        dataloader = DataLoader(
            dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True
        )
        print(f"Dataloader created with {len(dataloader)} batches.")
        mlflow.log_param("num_training_samples", len(dataset))
        mlflow.log_param("num_batches", len(dataloader))


        # --- Model Initialization ---
        gen_AB = UNetGenerator(in_channels=config['CHANNELS_IMG_GRAY'], out_channels=config['CHANNELS_IMG_COLOR']).to(DEVICE)
        gen_BA = UNetGenerator(in_channels=config['CHANNELS_IMG_COLOR'], out_channels=config['CHANNELS_IMG_GRAY']).to(DEVICE)
        disc_Y = Discriminator(in_channels_x=config['CHANNELS_IMG_GRAY'], in_channels_y=config['CHANNELS_IMG_COLOR']).to(DEVICE)
        disc_X = Discriminator(in_channels_x=config['CHANNELS_IMG_COLOR'], in_channels_y=config['CHANNELS_IMG_GRAY']).to(DEVICE)

        gen_AB.apply(weights_init); gen_BA.apply(weights_init)
        disc_X.apply(weights_init); disc_Y.apply(weights_init)

        vgg_content_loss_net = VGGContentLoss(feature_layer_index=config['CHOSEN_VGG_LAYER_INDEX'], device=DEVICE).to(DEVICE)
        vgg_content_loss_net.eval()

        # --- Optimizers ---
        opt_gen = optim.Adam(
            list(gen_AB.parameters()) + list(gen_BA.parameters()),
            lr=config['LEARNING_RATE_GEN'], betas=(0.5, 0.999)
        )
        opt_disc = optim.Adam(
            list(disc_X.parameters()) + list(disc_Y.parameters()),
            lr=config['LEARNING_RATE_DISC'], betas=(0.5, 0.999)
        )

        # --- Loss Functions ---
        bce_loss = nn.BCEWithLogitsLoss()
        l1_loss = nn.L1Loss()

        # --- Training Loop ---
        # Create output directories if they don't exist for local saving of examples/checkpoints
        os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
        os.makedirs(config['CHECKPOINT_DIR'], exist_ok=True)


        for epoch in range(config['NUM_EPOCHS']):
            loop = tqdm(dataloader, leave=True, desc=f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}]")
            current_G_loss_epoch_sum = 0
            current_D_loss_epoch_sum = 0
            num_batches_processed = 0

            for batch_idx, (real_A_gray, real_B_color) in enumerate(loop):
                real_A_gray = real_A_gray.to(DEVICE)
                real_B_color = real_B_color.to(DEVICE)

                # --- Train Discriminators ---
                for param in gen_AB.parameters(): param.requires_grad = False
                for param in gen_BA.parameters(): param.requires_grad = False
                for param in disc_X.parameters(): param.requires_grad = True
                for param in disc_Y.parameters(): param.requires_grad = True
                opt_disc.zero_grad(set_to_none=True)

                fake_B_color = gen_AB(real_A_gray)
                D_Y_real_pred = disc_Y(real_A_gray, real_B_color)
                loss_D_Y_real = bce_loss(D_Y_real_pred, torch.ones_like(D_Y_real_pred))
                D_Y_fake_pred = disc_Y(real_A_gray, fake_B_color.detach())
                loss_D_Y_fake = bce_loss(D_Y_fake_pred, torch.zeros_like(D_Y_fake_pred))
                loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5
                loss_D_Y.backward()

                fake_A_gray = gen_BA(real_B_color)
                D_X_real_pred = disc_X(real_B_color, real_A_gray)
                loss_D_X_real = bce_loss(D_X_real_pred, torch.ones_like(D_X_real_pred))
                D_X_fake_pred = disc_X(real_B_color, fake_A_gray.detach())
                loss_D_X_fake = bce_loss(D_X_fake_pred, torch.zeros_like(D_X_fake_pred))
                loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5
                loss_D_X.backward()
                opt_disc.step()
                
                total_loss_D_batch = loss_D_X + loss_D_Y
                current_D_loss_epoch_sum += total_loss_D_batch.item()

                # --- Train Generators ---
                for param in disc_X.parameters(): param.requires_grad = False
                for param in disc_Y.parameters(): param.requires_grad = False
                for param in gen_AB.parameters(): param.requires_grad = True
                for param in gen_BA.parameters(): param.requires_grad = True
                opt_gen.zero_grad(set_to_none=True)

                fake_B_color_for_G = gen_AB(real_A_gray)
                fake_A_gray_for_G = gen_BA(real_B_color)

                D_Y_fake_pred_for_G = disc_Y(real_A_gray, fake_B_color_for_G)
                loss_adv_G_AB = bce_loss(D_Y_fake_pred_for_G, torch.ones_like(D_Y_fake_pred_for_G))
                D_X_fake_pred_for_G = disc_X(real_B_color, fake_A_gray_for_G) # Corrected variable name
                loss_adv_G_BA = bce_loss(D_X_fake_pred_for_G, torch.ones_like(D_X_fake_pred_for_G)) # Corrected variable name

                cycled_A_gray = gen_BA(fake_B_color_for_G)
                loss_cycle_A = l1_loss(cycled_A_gray, real_A_gray) * config['LAMBDA_CYCLE']
                cycled_B_color = gen_AB(fake_A_gray_for_G)
                loss_cycle_B = l1_loss(cycled_B_color, real_B_color) * config['LAMBDA_CYCLE']
                
                loss_L1_G_AB = l1_loss(fake_B_color_for_G, real_B_color) * config['LAMBDA_L1']
                
                with torch.no_grad(): real_B_features = vgg_content_loss_net(real_B_color)
                fake_B_features = vgg_content_loss_net(fake_B_color_for_G)
                loss_content_G_AB = l1_loss(fake_B_features, real_B_features.detach()) * config['LAMBDA_CONTENT']
                
                loss_G_batch = loss_adv_G_AB + loss_adv_G_BA + loss_cycle_A + loss_cycle_B + loss_L1_G_AB + loss_content_G_AB
                loss_G_batch.backward()
                opt_gen.step()
                current_G_loss_epoch_sum += loss_G_batch.item()
                
                num_batches_processed += 1

                # Log batch metrics to MLflow
                if batch_idx % 20 == 0: # Log every 20 batches to avoid too much I/O
                    mlflow.log_metrics({
                        "batch_loss_D_X": loss_D_X.item(), "batch_loss_D_Y": loss_D_Y.item(),
                        "batch_total_D_Loss": total_loss_D_batch.item(),
                        "batch_loss_G_Adv_AB": loss_adv_G_AB.item(), "batch_loss_G_Adv_BA": loss_adv_G_BA.item(),
                        "batch_loss_G_L1_AB": loss_L1_G_AB.item(), "batch_loss_G_Content_AB": loss_content_G_AB.item(),
                        "batch_loss_G_Cycle_A": loss_cycle_A.item(), "batch_loss_G_Cycle_B": loss_cycle_B.item(),
                        "batch_total_G_Loss": loss_G_batch.item()
                    }, step=epoch * len(dataloader) + batch_idx)
                
                loop.set_postfix(D_loss=f"{total_loss_D_batch.item():.3f}", G_loss=f"{loss_G_batch.item():.3f}")
                
                # Clean up to free memory
                del real_A_gray, real_B_color, fake_A_gray, fake_B_color, cycled_A_gray, cycled_B_color
                del fake_A_gray_for_G, fake_B_color_for_G
                del D_Y_real_pred, D_Y_fake_pred, D_X_real_pred, D_X_fake_pred
                del D_Y_fake_pred_for_G, D_X_fake_pred_for_G # Corrected variable
                if DEVICE == "cuda": torch.cuda.empty_cache()

            # End of Epoch
            avg_G_loss_this_epoch = current_G_loss_epoch_sum / num_batches_processed if num_batches_processed > 0 else 0
            avg_D_loss_this_epoch = current_D_loss_epoch_sum / num_batches_processed if num_batches_processed > 0 else 0

            mlflow.log_metrics({
                "epoch_avg_G_Loss": avg_G_loss_this_epoch,
                "epoch_avg_D_Loss": avg_D_loss_this_epoch,
            }, step=epoch)
            print(f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}] - Avg G Loss: {avg_G_loss_this_epoch:.4f}, Avg D Loss: {avg_D_loss_this_epoch:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == config['NUM_EPOCHS'] - 1:
                save_some_examples(
                    gen_AB, dataloader, epoch,
                    output_folder=os.path.join(config['OUTPUT_DIR'], f"epoch_{epoch+1}_samples"),
                    current_device=DEVICE,
                    log_to_wandb_or_mlflow=mlflow, # Pass mlflow client
                    batch_size_display=config['BATCH_SIZE'] # Or a fixed number like 1 or 2
                )
                
                # Log models with MLflow
                # Note: For CycleGAN, you have multiple models. You might log them separately
                # or as a dictionary of models.
                # For the Streamlit app, we'll primarily care about gen_AB.
                if (epoch + 1) % 10 == 0 or epoch == config['NUM_EPOCHS'] - 1: # Save models less frequently
                    mlflow.pytorch.log_model(gen_AB, "generator_AB", registered_model_name="SAR_Generator_AB")
                    mlflow.pytorch.log_model(gen_BA, "generator_BA") # Not registering this one for now
                    mlflow.pytorch.log_model(disc_X, "discriminator_X")
                    mlflow.pytorch.log_model(disc_Y, "discriminator_Y")
                    print(f"Models logged to MLflow for epoch {epoch+1}")

            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()
        
        mlflow.log_metric("status", 1) # 1 for success
        print("Training finished.")

if __name__ == "__main__":
    main()
