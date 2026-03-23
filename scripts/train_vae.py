import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("GPU", help="GPU ID 0,1..", type=int)
parser.add_argument("DIM", help="Latent dims", type=int)
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
from library import *
import math
import torch.nn as nn
from datetime import datetime
import shutil
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from library.data_handling.hr_patches_dataset import fetch_hr_patches_dataloaders
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

EXPERIMENT_TAG = 'ImageVAE_L2_MS_SSIM'
BATCH_SIZE = 64
LR_START = 5e-4
LR_END = 1e-6
EPOCHS = 250
LOSS = 'MSE'
MS_SSIM_WEIGHT = 0.15
BETA_FINAL = 0.0015
BETA_FINAL_EPOCH = 75
DATA_PATH = '/data/ramsookd/DATASET_HR'
MODEL_CONFIG = 'configs/models/image_vae.toml'
DEVICE = 'cuda'
STARTING_MODEL = None

model_configs = load_toml_file(MODEL_CONFIG)
model_tag = f'{args.DIM}-channels'
model_args = model_configs[model_tag]

def get_beta(epoch: int) -> float:
    if BETA_FINAL_EPOCH <= 0:
        return BETA_FINAL
    t = min(epoch / BETA_FINAL_EPOCH, 1.0)
    return BETA_FINAL * 0.5 * (1.0 - math.cos(math.pi * t))

def kl_divergence_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

if __name__ == '__main__':
    results_folder = os.path.join('results', EXPERIMENT_TAG, model_tag, datetime.now().strftime("%d - %m - %Y, %H:%M:%S"))
    logs_folder = os.path.join(results_folder, 'logs')
    model_folder = os.path.join(results_folder, 'models')
    for _path in (results_folder, logs_folder, model_folder):
        os.makedirs(_path, exist_ok=True)
    shutil.copyfile(MODEL_CONFIG, os.path.join(results_folder, os.path.basename(MODEL_CONFIG)))

    train_dataloader, test_dataloader = fetch_hr_patches_dataloaders(DATA_PATH, BATCH_SIZE, 3)

    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    elif LOSS == 'MAE':
        criterion = nn.L1Loss()

    use_ms_ssim = MS_SSIM_WEIGHT > 0.0
    if use_ms_ssim:
        msssim_criterion = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        msssim_criterion.to(DEVICE)
    criterion.to(DEVICE)

    model = ImageVAE(**model_args)
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
        beta = get_beta(epoch)
        train_losses = []
        train_mse_losses = []
        train_kl_losses = []
        train_ms_ssim_vals = []
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=False)
        for train_step, train_data in enumerate(pbar):
            reference = train_data
            reference = reference.to(DEVICE)
            optimizer.zero_grad()
            mu, log_var = model.encode(reference)
            z = model.sampling(mu, log_var)
            predicted = model.decode(z)
            mse_loss = criterion(predicted, reference)
            kl_loss = kl_divergence_loss(mu, log_var)
            if use_ms_ssim:
                msssim_val = msssim_criterion(predicted, reference)
                ms_ssim_term = (-1.0 * msssim_val) * MS_SSIM_WEIGHT
                train_ms_ssim_vals.append(msssim_val.item())
            else:
                ms_ssim_term = 0.0
            loss = mse_loss + beta * kl_loss + ms_ssim_term
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            train_losses.append(loss_val)
            train_mse_losses.append(mse_loss.item())
            train_kl_losses.append(kl_loss.item())
            pbar.set_postfix(loss=f'{loss_val:.6f}', avg_loss=f'{sum(train_losses) / len(train_losses):.6f}')

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_mse = sum(train_mse_losses) / len(train_mse_losses)
        avg_train_kl = sum(train_kl_losses) / len(train_kl_losses)
        writer.add_scalar('loss/train', avg_train_loss, epoch)
        writer.add_scalar('loss/train_mse', avg_train_mse, epoch)
        writer.add_scalar('loss/train_kl', avg_train_kl, epoch)
        writer.add_scalar('beta', beta, epoch)
        if use_ms_ssim:
            avg_train_ms_ssim = sum(train_ms_ssim_vals) / len(train_ms_ssim_vals)
            writer.add_scalar('ms_ssim/train', avg_train_ms_ssim, epoch)

        scheduler.step()

        model.eval()
        test_losses = []
        test_mse_losses = []
        test_kl_losses = []
        test_ms_ssim_vals = []
        for test_step, test_data in enumerate(test_dataloader):
            reference = test_data
            reference = reference.to(DEVICE)
            with torch.no_grad():
                mu, log_var = model.encode(reference)
                z = model.sampling(mu, log_var)
                predicted = model.decode(z)
                mse_loss = criterion(predicted, reference)
                kl_loss = kl_divergence_loss(mu, log_var)
                if use_ms_ssim:
                    msssim_val = msssim_criterion(predicted, reference)
                    ms_ssim_term = (-1.0 * msssim_val) * MS_SSIM_WEIGHT
                    test_ms_ssim_vals.append(msssim_val.item())
                else:
                    ms_ssim_term = 0.0
                loss = mse_loss + beta * kl_loss + ms_ssim_term
            test_losses.append(loss.item())
            test_mse_losses.append(mse_loss.item())
            test_kl_losses.append(kl_loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_mse = sum(test_mse_losses) / len(test_mse_losses)
        avg_test_kl = sum(test_kl_losses) / len(test_kl_losses)
        writer.add_scalar('loss/test', avg_test_loss, epoch)
        writer.add_scalar('loss/test_mse', avg_test_mse, epoch)
        writer.add_scalar('loss/test_kl', avg_test_kl, epoch)
        if use_ms_ssim:
            avg_test_ms_ssim = sum(test_ms_ssim_vals) / len(test_ms_ssim_vals)
            writer.add_scalar('ms_ssim/test', avg_test_ms_ssim, epoch)

        torch.save(model.state_dict(), os.path.join(model_folder, f'epoch_{epoch:04d}.pt'))

    writer.close()
