from library import *
import os
import torch.nn as nn
from datetime import datetime
import shutil
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

EXPERIMENT_TAG = 'trini_base_allQPs'
BATCH_SIZE = 64
LR_START = 5e-4
LR_END = 1e-6
EPOCHS = 1000
LOSS = 'MSE'
DATA_PATH = '/storage/8TB-SSD-1/DATASET'
QPS = [51]
PATCH_SIZE = 64
MODEL_CONFIG = 'configs/models/trini_base.toml'
DEVICE = 'cuda'

if __name__ == '__main__':
    results_folder = os.path.join('results', EXPERIMENT_TAG, datetime.now().strftime("%d - %m - %Y, %H:%M:%S"))
    logs_folder = os.path.join(results_folder, 'logs')
    model_folder = os.path.join(results_folder, 'models')
    for _path in (results_folder, logs_folder, model_folder):
        os.makedirs(_path, exist_ok=True)
    shutil.copyfile(MODEL_CONFIG, os.path.join(results_folder, os.path.basename(MODEL_CONFIG)))

    train_dataloader, test_dataloader = fetch_video_dataloaders(DATA_PATH, BATCH_SIZE, QPS, PATCH_SIZE)

    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    elif LOSS == 'MAE':
        criterion = nn.L1Loss()

    criterion.to(DEVICE)

    model_args = load_toml_file(MODEL_CONFIG)
    model = TRINIModel(**model_args)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR_END)

    writer = SummaryWriter(log_dir=logs_folder)

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=False)
        for train_step, train_data in enumerate(pbar):
            compressed, reference = train_data
            compressed = compressed.to(DEVICE)
            reference = reference.to(DEVICE)
            optimizer.zero_grad()
            enhanced = model(reference, compressed)
            loss = criterion(enhanced, reference)
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
            compressed, reference = test_data
            compressed = compressed.to(DEVICE)
            reference = reference.to(DEVICE)
            with torch.no_grad():
                enhanced = model(reference, compressed)
                loss = criterion(enhanced, reference)
            test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)
        writer.add_scalar('loss/test', avg_test_loss, epoch)

        torch.save(model.state_dict(), os.path.join(model_folder, f'epoch_{epoch:04d}.pt'))

    writer.close()
