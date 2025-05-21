# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import gc
import yaml 
import mlflow 
import mlflow.pytorch
import matplotlib.pyplot as plt # For final loss plot

from model import UNetGenerator, Discriminator, VGGContentLoss, weights_init
from dataloader import SentinelDataset, get_transforms
from utils import save_some_examples, set_seed

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    set_seed(config['MANUAL_SEED'])

    DEVICE = config['DEVICE'] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    if DEVICE == "cpu":
        print("Warning: CUDA not available, training on CPU will be very slow.")

    mlflow.set_tracking_uri(config['MLFLOW_TRACKING_URI'])
    experiment_name = config['MLFLOW_EXPERIMENT_NAME']
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"MLflow experiment '{experiment_name}' created with ID: {experiment_id}")
        except mlflow.exceptions.MlflowException as e: # Handle race condition or other creation errors
             print(f"Error creating MLflow experiment: {e}. Trying to get existing one.")
             experiment = mlflow.get_experiment_by_name(experiment_name)
             if not experiment:
                 raise ValueError(f"Could not create or find MLflow experiment '{experiment_name}'")
             experiment_id = experiment.experiment_id
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing MLflow experiment '{experiment_name}' with ID: {experiment_id}")
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="CycleGAN_SAR_Training_Run") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.log_params(config)
        
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param("git_commit_hash", repo.head.object.hexsha)
            mlflow.log_artifact("config/config.yaml", artifact_path="run_config")
        except Exception as e: # Catch generic exception for git issues
            print(f"Git info logging skipped: {e}")

        data_source_path = config['DATA_DIR']

        transform_gray, transform_color = get_transforms(config['IMG_SIZE'])
        dataset = SentinelDataset(
            root_dir=data_source_path,
            terrain_classes=config['TERRAIN_CLASSES_TO_USE'],
            transform_gray=transform_gray,
            transform_color=transform_color,
            img_size=config['IMG_SIZE'],
            max_images_per_class=config.get('MAX_IMAGES_PER_CLASS', 1000),
            gcs_bucket_name=config.get('GCS_BUCKET_NAME') if data_source_path.startswith("gs://") else None,
            gcp_project_id=config.get('GCP_PROJECT_ID')
        )

        if len(dataset) == 0:
            print("CRITICAL: Dataset is empty. Halting execution.")
            mlflow.log_metric("status_completion", 0) 
            mlflow.set_tag("mlflow.runName", f"{run.data.tags.get('mlflow.runName', 'Run')} - FAILED (No Data)")
            return

        dataloader = DataLoader(
            dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
            num_workers=config.get('DATALOADER_WORKERS', 2), pin_memory=True, drop_last=True
        )
        print(f"Dataloader created with {len(dataloader)} batches from {len(dataset)} samples.")
        mlflow.log_param("num_training_samples", len(dataset))
        mlflow.log_param("num_batches", len(dataloader))

        gen_AB = UNetGenerator(in_channels=config['CHANNELS_IMG_GRAY'], out_channels=config['CHANNELS_IMG_COLOR']).to(DEVICE)
        gen_BA = UNetGenerator(in_channels=config['CHANNELS_IMG_COLOR'], out_channels=config['CHANNELS_IMG_GRAY']).to(DEVICE)
        disc_Y = Discriminator(in_channels_x=config['CHANNELS_IMG_GRAY'], in_channels_y=config['CHANNELS_IMG_COLOR']).to(DEVICE)
        disc_X = Discriminator(in_channels_x=config['CHANNELS_IMG_COLOR'], in_channels_y=config['CHANNELS_IMG_GRAY']).to(DEVICE)

        gen_AB.apply(weights_init); gen_BA.apply(weights_init)
        disc_X.apply(weights_init); disc_Y.apply(weights_init)

        vgg_content_loss_net = VGGContentLoss(feature_layer_index=config['CHOSEN_VGG_LAYER_INDEX'], device=DEVICE).to(DEVICE)
        vgg_content_loss_net.eval()

        opt_gen = optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=config['LEARNING_RATE_GEN'], betas=(0.5, 0.999))
        opt_disc = optim.Adam(list(disc_X.parameters()) + list(disc_Y.parameters()), lr=config['LEARNING_RATE_DISC'], betas=(0.5, 0.999))

        bce_loss = nn.BCEWithLogitsLoss()
        l1_loss = nn.L1Loss()

        local_output_dir = config.get('OUTPUT_DIR_LOCAL', "./outputs/local_run_output")
        os.makedirs(local_output_dir, exist_ok=True)

        g_losses_epoch_avg_list = []
        d_losses_epoch_avg_list = []

        print(f"Starting training for {config['NUM_EPOCHS']} epochs...")
        for epoch in range(config['NUM_EPOCHS']):
            loop = tqdm(dataloader, leave=True, desc=f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}]")
            current_G_loss_epoch_sum = 0.0
            current_D_loss_epoch_sum = 0.0
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

                with torch.no_grad(): # gen_AB/gen_BA are not training here
                    fake_B_color_detached = gen_AB(real_A_gray)
                    fake_A_gray_detached = gen_BA(real_B_color)

                D_Y_real_pred = disc_Y(real_A_gray, real_B_color)
                loss_D_Y_real = bce_loss(D_Y_real_pred, torch.ones_like(D_Y_real_pred).to(DEVICE))
                D_Y_fake_pred = disc_Y(real_A_gray, fake_B_color_detached) # Use detached version
                loss_D_Y_fake = bce_loss(D_Y_fake_pred, torch.zeros_like(D_Y_fake_pred).to(DEVICE))
                loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5
                loss_D_Y.backward()

                D_X_real_pred = disc_X(real_B_color, real_A_gray)
                loss_D_X_real = bce_loss(D_X_real_pred, torch.ones_like(D_X_real_pred).to(DEVICE))
                D_X_fake_pred = disc_X(real_B_color, fake_A_gray_detached) # Use detached version
                loss_D_X_fake = bce_loss(D_X_fake_pred, torch.zeros_like(D_X_fake_pred).to(DEVICE))
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

                fake_B_color_for_G = gen_AB(real_A_gray) # For G_AB losses
                fake_A_gray_for_G = gen_BA(real_B_color) # For G_BA losses

                D_Y_fake_pred_for_G = disc_Y(real_A_gray, fake_B_color_for_G)
                loss_adv_G_AB = bce_loss(D_Y_fake_pred_for_G, torch.ones_like(D_Y_fake_pred_for_G).to(DEVICE))
                D_X_fake_pred_for_G = disc_X(real_B_color, fake_A_gray_for_G)
                loss_adv_G_BA = bce_loss(D_X_fake_pred_for_G, torch.ones_like(D_X_fake_pred_for_G).to(DEVICE))

                cycled_A_gray = gen_BA(fake_B_color_for_G)
                loss_cycle_A = l1_loss(cycled_A_gray, real_A_gray) * config['LAMBDA_CYCLE']
                cycled_B_color = gen_AB(fake_A_gray_for_G)
                loss_cycle_B = l1_loss(cycled_B_color, real_B_color) * config['LAMBDA_CYCLE']
                
                loss_L1_G_AB = l1_loss(fake_B_color_for_G, real_B_color) * config['LAMBDA_L1']
                
                with torch.no_grad(): real_B_features = vgg_content_loss_net(real_B_color)
                fake_B_features = vgg_content_loss_net(fake_B_color_for_G)
                loss_content_G_AB = l1_loss(fake_B_features, real_B_features.detach()) * config['LAMBDA_CONTENT']
                
                loss_G_batch = (loss_adv_G_AB + loss_adv_G_BA + 
                                loss_cycle_A + loss_cycle_B + 
                                loss_L1_G_AB + loss_content_G_AB)
                loss_G_batch.backward()
                opt_gen.step()
                current_G_loss_epoch_sum += loss_G_batch.item()
                
                num_batches_processed += 1

                if batch_idx % config.get("MLFLOW_BATCH_LOG_INTERVAL", 50) == 0:
                    metrics_to_log = {
                        "batch_loss_D_X": loss_D_X.item(), "batch_loss_D_Y": loss_D_Y.item(),
                        "batch_total_D_Loss": total_loss_D_batch.item(),
                        "batch_loss_G_Adv_AB": loss_adv_G_AB.item(), "batch_loss_G_Adv_BA": loss_adv_G_BA.item(),
                        "batch_loss_G_L1_AB": loss_L1_G_AB.item(), 
                        "batch_loss_G_Content_AB": loss_content_G_AB.item(),
                        "batch_loss_G_Cycle_A": loss_cycle_A.item(), "batch_loss_G_Cycle_B": loss_cycle_B.item(),
                        "batch_total_G_Loss": loss_G_batch.item()
                    }
                    mlflow.log_metrics(metrics_to_log, step=epoch * len(dataloader) + batch_idx)
                
                loop.set_postfix(D_loss=f"{total_loss_D_batch.item():.3f}", G_loss=f"{loss_G_batch.item():.3f}")
                
                del real_A_gray, real_B_color, fake_B_color_detached, fake_A_gray_detached
                del fake_A_gray_for_G, fake_B_color_for_G, cycled_A_gray, cycled_B_color
                del D_Y_real_pred, D_Y_fake_pred, D_X_real_pred, D_X_fake_pred
                del D_Y_fake_pred_for_G, D_X_fake_pred_for_G, real_B_features, fake_B_features
                if DEVICE == "cuda": torch.cuda.empty_cache()

            avg_G_loss_this_epoch = current_G_loss_epoch_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
            avg_D_loss_this_epoch = current_D_loss_epoch_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
            
            g_losses_epoch_avg_list.append(avg_G_loss_this_epoch)
            d_losses_epoch_avg_list.append(avg_D_loss_this_epoch)

            mlflow.log_metrics({"epoch_avg_G_Loss": avg_G_loss_this_epoch, "epoch_avg_D_Loss": avg_D_loss_this_epoch}, step=epoch)
            print(f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}] - Avg G Loss: {avg_G_loss_this_epoch:.4f}, Avg D Loss: {avg_D_loss_this_epoch:.4f}")

            if (epoch + 1) % config.get("SAVE_EXAMPLE_EPOCH_INTERVAL", 5) == 0 or epoch == config['NUM_EPOCHS'] - 1:
                example_local_folder = os.path.join(local_output_dir, f"epoch_{epoch+1}_samples")
                save_some_examples(
                    gen_AB, dataloader, epoch,
                    output_folder_local=example_local_folder,
                    current_device=DEVICE,
                    mlflow_client=mlflow, 
                    batch_size_display=config.get('EXAMPLE_BATCH_SIZE_DISPLAY', 1)
                )
            
            if (epoch + 1) % config.get("SAVE_MODEL_EPOCH_INTERVAL", 10) == 0 or epoch == config['NUM_EPOCHS'] - 1:
                print(f"Logging models to MLflow for epoch {epoch+1}...")
                mlflow.pytorch.log_model(pytorch_model=gen_AB, artifact_path="generator_AB_model", registered_model_name=config.get("MLFLOW_MODEL_NAME_GEN_AB"))
                mlflow.pytorch.log_state_dict(gen_AB.state_dict(), artifact_path="model_state_dict")
                mlflow.pytorch.log_model(gen_BA, "generator_BA_model") 
                mlflow.pytorch.log_model(disc_X, "discriminator_X_model")
                mlflow.pytorch.log_model(disc_Y, "discriminator_Y_model")
                print(f"Models logged for epoch {epoch+1}.")

            gc.collect(); 
            if DEVICE == "cuda": torch.cuda.empty_cache()
        
        if g_losses_epoch_avg_list and d_losses_epoch_avg_list:
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Avg Loss per Epoch")
            plt.plot(g_losses_epoch_avg_list, label="Avg G Loss")
            plt.plot(d_losses_epoch_avg_list, label="Avg D Loss")
            plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()
            local_loss_plot_path = os.path.join(local_output_dir, "final_loss_plot_avg_epoch.png")
            plt.savefig(local_loss_plot_path)
            plt.close()
            mlflow.log_artifact(local_loss_plot_path, artifact_path="training_plots")
            print(f"Final loss plot saved to {local_loss_plot_path} and logged to MLflow.")

        mlflow.log_metric("status_completion", 1)
        mlflow.set_tag("mlflow.runName", f"{run.data.tags.get('mlflow.runName', 'Run')} - COMPLETED")
        print("Training finished successfully and logged to MLflow.")

if __name__ == "__main__":
    main()
