import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import SDFNet
from loader import SDFData
from renderer import plot_sdf

TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
MODEL_PATH = '../models/'
RES_PATH = '../results/trained_heatmaps/'
MASK_PATH = '../shapes/masks/'
LOG_PATH = '../logs/'

if __name__ == '__main__':
    batch_size = 64
    learning_rate = 1e-5
    epochs = 1000
    regularization = 0  # Default: 1e-2
    delta = 0.1  # Truncated distance

    print('Enter shape name:')
    name = input()

    train_data = SDFData(f'{TRAIN_DATA_PATH}{name}.txt')
    val_data = SDFData(f'{VAL_DATA_PATH}{name}.txt')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    model = SDFNet().to(device)
    if os.path.exists(f'{MODEL_PATH}{name}.pth'):
        model.load_state_dict(torch.load(f'{MODEL_PATH}{name}.pth'))

    loss_fn = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)

    writer = SummaryWriter(LOG_PATH)
    total_train_step = 0
    total_val_step = 0

    start_time = time.time()
    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')

        # Training loop
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (xy, sdf) in enumerate(train_dataloader):
            xy, sdf = xy.to(device), sdf.to(device)
            pred_sdf = model(xy)
            sdf = torch.reshape(sdf, pred_sdf.shape)
            loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(sdf, min=-delta, max=delta))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 50 == 0:
                loss_value, current = loss.item(), batch * len(xy)
                print(f'loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]')

            total_train_step += 1
            if total_train_step % 200 == 0:
                writer.add_scalar('Training loss', loss.item(), total_train_step)

        # Evaluation loop
        model.eval()
        size = len(val_dataloader.dataset)
        val_loss = 0

        with torch.no_grad():
            for xy, sdf in val_dataloader:
                xy, sdf = xy.to(device), sdf.to(device)
                pred_sdf = model(xy)
                sdf = torch.reshape(sdf, pred_sdf.shape)
                loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(sdf, min=-delta, max=delta))
                val_loss += loss

        val_loss /= size
        end_time = time.time()
        print(f'Test Error: \n Avg loss: {val_loss:>8f} \n Time: {(end_time - start_time):>12f} \n ')

        total_val_step += 1
        writer.add_scalar('Val loss', val_loss, total_val_step)

    torch.save(model.state_dict(), f'{MODEL_PATH}{name}.pth')
    print(f'Complete training with {epochs} epochs!')

    writer.close()

    # Plot results
    print('Plotting results...')
    plot_sdf(model, device, res_path=RES_PATH, name=name, mask_path=MASK_PATH, is_net=True, show=False)
    print('Done!')
