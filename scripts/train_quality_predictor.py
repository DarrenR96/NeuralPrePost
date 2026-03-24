from library import *
import os
import torch.nn as nn
from datetime import datetime
import shutil
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from library.data_handling.qualiy_prediction_dataset import fetch_qualtiy_prediction_dataloaders

EXPERIMENT_TAG = 'QualityPredictor_BASE'
VAE_MODEL = 'savedModels/ImageVAE'
BATCH_SIZE = 32
LR_START = 1e-4
LR_END = 1e-6
EPOCHS = 250
LOSS = 'MSE'
DATA_PATH = '/storage/8TB-SSD-1/DATASET_1080p'
MODEL_CONFIG = 'configs/models/quality_predictor_base.toml'
DEVICE = 'cuda'
STARTING_MODEL = None


@torch.no_grad()
def blended_vae_latent_means(
    vae: ImageVAE,
    left_frame: torch.Tensor,
    right_frame: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
) -> torch.Tensor:
    mu_left, _ = vae.encode(left_frame)
    mu_right, _ = vae.encode(right_frame)
    ls = left_scale.unsqueeze(-1).unsqueeze(-1)
    rs = right_scale.unsqueeze(-1).unsqueeze(-1)
    return ls * mu_left + rs * mu_right


if __name__ == '__main__':
    results_folder = os.path.join('results', EXPERIMENT_TAG, datetime.now().strftime("%d - %m - %Y, %H:%M:%S"))
    logs_folder = os.path.join(results_folder, 'logs')
    model_folder = os.path.join(results_folder, 'models')
    for _path in (results_folder, logs_folder, model_folder):
        os.makedirs(_path, exist_ok=True)
    shutil.copyfile(MODEL_CONFIG, os.path.join(results_folder, os.path.basename(MODEL_CONFIG)))
    shutil.copyfile(
        os.path.join(VAE_MODEL, 'config.toml'),
        os.path.join(results_folder, 'vae_config.toml'),
    )

    train_dataloader, test_dataloader = fetch_qualtiy_prediction_dataloaders(DATA_PATH, BATCH_SIZE)

    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    elif LOSS == 'MAE':
        criterion = nn.L1Loss()

    criterion.to(DEVICE)

    vae_model = load_torch_model(VAE_MODEL, ImageVAE)
    vae_model.to(DEVICE)
    vae_model.eval()
    for _p in vae_model.parameters():
        _p.requires_grad = False

    model_args = load_toml_file(MODEL_CONFIG)
    model = LatentQualityPredictorNetwork(**model_args)
    model.to(DEVICE)

    if STARTING_MODEL is not None:
        state = torch.load(STARTING_MODEL, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        print(f'Loaded weights from {STARTING_MODEL} for fine-tuning.')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR_END)

    writer = SummaryWriter(log_dir=logs_folder)

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=False)
        for train_step, train_data in enumerate(pbar):
            left_frame, right_frame, left_scale, right_scale, target = train_data
            left_frame = left_frame.to(DEVICE)
            right_frame = right_frame.to(DEVICE)
            left_scale = left_scale.to(DEVICE)
            right_scale = right_scale.to(DEVICE)
            target = target.to(DEVICE)
            blended_latent = blended_vae_latent_means(
                vae_model, left_frame, right_frame, left_scale, right_scale
            )
            optimizer.zero_grad()
            predicted = model(blended_latent)
            loss = criterion(predicted, target)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            train_losses.append(loss_val)
            pbar.set_postfix(loss=f'{loss_val:.6f}', avg_loss=f'{sum(train_losses) / len(train_losses):.6f}')

        avg_train_loss = sum(train_losses) / len(train_losses)
        writer.add_scalar('loss/train', avg_train_loss, epoch)

        scheduler.step()

        model.eval()
        test_losses = []
        for test_step, test_data in enumerate(test_dataloader):
            left_frame, right_frame, left_scale, right_scale, target = test_data
            left_frame = left_frame.to(DEVICE)
            right_frame = right_frame.to(DEVICE)
            left_scale = left_scale.to(DEVICE)
            right_scale = right_scale.to(DEVICE)
            target = target.to(DEVICE)
            blended_latent = blended_vae_latent_means(
                vae_model, left_frame, right_frame, left_scale, right_scale
            )
            with torch.no_grad():
                predicted = model(blended_latent)
                loss = criterion(predicted, target)
            test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)
        writer.add_scalar('loss/test', avg_test_loss, epoch)

        torch.save(model.state_dict(), os.path.join(model_folder, f'epoch_{epoch:04d}.pt'))

    writer.close()
