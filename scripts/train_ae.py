from library import *
import os
import torch.nn as nn
from datetime import datetime
import shutil
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from library.data_handling.hr_patches_dataset import fetch_hr_patches_dataloaders
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

EXPERIMENT_TAG = 'ImageAE_BASE_MSSIM'
BATCH_SIZE = 64
LR_START = 1e-4
LR_END = 1e-6
EPOCHS = 250
LOSS = 'MAE'
MS_SSIM_WEIGHT = 0.15
DATA_PATH = '/storage/8TB-SSD-1/DATASET_HR'
MODEL_CONFIG = 'configs/models/image_ae_base.toml'
DEVICE = 'cuda'
STARTING_MODEL = 'results/ImageAE_BASE/13 - 03 - 2026, 15:14:35/models/epoch_0249.pt'

if __name__ == '__main__':
    results_folder = os.path.join('results', EXPERIMENT_TAG, datetime.now().strftime("%d - %m - %Y, %H:%M:%S"))
    logs_folder = os.path.join(results_folder, 'logs')
    model_folder = os.path.join(results_folder, 'models')
    for _path in (results_folder, logs_folder, model_folder):
        os.makedirs(_path, exist_ok=True)
    shutil.copyfile(MODEL_CONFIG, os.path.join(results_folder, os.path.basename(MODEL_CONFIG)))

    train_dataloader, test_dataloader = fetch_hr_patches_dataloaders(DATA_PATH, BATCH_SIZE)

    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    elif LOSS == 'MAE':
        criterion = nn.L1Loss()

    msssim_criterion = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    msssim_criterion.to(DEVICE)
    criterion.to(DEVICE)

    model_args = load_toml_file(MODEL_CONFIG)
    model = ImageAE(**model_args)
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
        train_mse_losses = []
        train_ms_ssim_vals = []
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=False)
        for train_step, train_data in enumerate(pbar):
            reference = train_data
            reference = reference.to(DEVICE)
            optimizer.zero_grad()
            predicted = model(reference)
            msssim_val = msssim_criterion(predicted, reference)
            mse_loss = criterion(predicted, reference)
            loss = mse_loss + (-1.0 * msssim_val) * MS_SSIM_WEIGHT
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            train_losses.append(loss_val)
            train_mse_losses.append(mse_loss.item())
            train_ms_ssim_vals.append(msssim_val.item())
            pbar.set_postfix(loss=f'{loss_val:.6f}', avg_loss=f'{sum(train_losses) / len(train_losses):.6f}')

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_mse = sum(train_mse_losses) / len(train_mse_losses)
        avg_train_ms_ssim = sum(train_ms_ssim_vals) / len(train_ms_ssim_vals)
        writer.add_scalar('loss/train', avg_train_loss, epoch)
        writer.add_scalar('loss/train_mse', avg_train_mse, epoch)
        writer.add_scalar('ms_ssim/train', avg_train_ms_ssim, epoch)

        scheduler.step()

        model.eval()
        test_losses = []
        test_mse_losses = []
        test_ms_ssim_vals = []
        for test_step, test_data in enumerate(test_dataloader):
            reference = test_data
            reference = reference.to(DEVICE)
            with torch.no_grad():
                predicted = model(reference)
                mse_loss = criterion(predicted, reference)
                msssim_val = msssim_criterion(predicted, reference)
                loss = mse_loss + (-1.0 * msssim_val) * MS_SSIM_WEIGHT
            test_losses.append(loss.item())
            test_mse_losses.append(mse_loss.item())
            test_ms_ssim_vals.append(msssim_val.item())

        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_mse = sum(test_mse_losses) / len(test_mse_losses)
        avg_test_ms_ssim = sum(test_ms_ssim_vals) / len(test_ms_ssim_vals)
        writer.add_scalar('loss/test', avg_test_loss, epoch)
        writer.add_scalar('loss/test_mse', avg_test_mse, epoch)
        writer.add_scalar('ms_ssim/test', avg_test_ms_ssim, epoch)

        torch.save(model.state_dict(), os.path.join(model_folder, f'epoch_{epoch:04d}.pt'))

    writer.close()
