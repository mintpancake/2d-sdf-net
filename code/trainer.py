import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from net import SDFNet
from loader import SDFData

TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
MODEL_PATH = '../models/'


def train_loop(dataloader, model, loss_fn, optimizer, device, theta=0.1):
    model.train()
    size = len(dataloader.dataset)
    for batch, (xy, sdf) in enumerate(dataloader):
        xy, sdf = xy.to(device), sdf.to(device)

        pred_sdf = model(xy)
        loss = loss_fn(torch.clamp(pred_sdf, min=-theta, max=theta), torch.clamp(sdf, min=-theta, max=theta))

        optimizer.zero_grad()
        loss.backward()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(xy)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def val_loop(dataloader, model, loss_fn, device, theta=0.1):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0

    with torch.no_grad():
        for xy, sdf in dataloader:
            xy, sdf = xy.to(device), sdf.to(device)
            pred_sdf = model(xy)
            loss = loss_fn(torch.clamp(pred_sdf, min=-theta, max=theta), torch.clamp(sdf, min=-theta, max=theta))
            test_loss += loss

    test_loss /= size
    print(f'Test Error: \n Avg loss: {test_loss:>8f} \n')


if __name__ == '__main__':
    batch_size = 64
    learning_rate = 1e-5
    epochs = 1000

    print('Enter shape name:')
    name = input()

    train_data = SDFData(f'{TRAIN_DATA_PATH}{name}.txt')
    val_data = SDFData(f'{VAL_DATA_PATH}{name}.txt')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SDFNet().to(device)
    if os.path.exists(f'{MODEL_PATH}{name}.pth'):
        model.load_state_dict(torch.load(f'{MODEL_PATH}{name}.pth'))

    loss_fn = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        val_loop(val_dataloader, model, loss_fn, device)
    torch.save(model.state_dict(), f'{MODEL_PATH}{name}.pth')
    print('Done!')
