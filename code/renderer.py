import os
import numpy as np
import cv2
import torch
from net import SDFNet
from shape import Polygon

MODEL_PATH = '../models/'
DATA_PATH = '../shapes/normalized/'
MASK_PATH = '../shapes/masks/'
TRAINED_PATH = '../results/trained_heatmaps/'
TRUE_PATH = '../results/true_heatmaps/'


# Adapted from https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
def plot_sdf(sdf_func, device, res_path, name, mask_path,
             img_size=800, is_net=False, show=False):
    # Sample the 2D domain as a regular grid
    low = -0.5
    high = 0.5
    grid_size = 100
    margin = 8e-3
    max_norm = 0.3  # Normalizing distance

    grid = np.linspace(low, high, grid_size + 1)
    if not is_net:
        sdf_map = [[sdf_func(np.float_([x_, y_]))
                    for x_ in grid] for y_ in grid]
        sdf_map = np.array(sdf_map, dtype=np.float64)
    else:
        # Input shape is [1, 2]
        sdf_func.eval()
        with torch.no_grad():
            sdf_map = [[sdf_func(torch.Tensor([[x_, y_]]).to(device)).detach().cpu()
                        for x_ in grid] for y_ in grid]
        sdf_map = torch.Tensor(sdf_map).cpu().numpy()

    sdf_map = sdf_map[:-1, :-1]
    max_norm = np.max(np.abs(sdf_map)) if max_norm == 0 else max_norm
    heat_map = sdf_map / max_norm * 127.5 + 127.5
    heat_map = np.minimum(heat_map, 255)
    heat_map = np.maximum(heat_map, 0)

    # Plot predicted boundary
    low_pos = sdf_map > -margin
    high_pos = sdf_map < margin
    edge_pos = low_pos & high_pos
    heat_map = np.where(edge_pos, 0, heat_map)

    # Scale to canvas size
    scale = int(img_size / grid_size)
    heat_map = np.kron(heat_map, np.ones((scale, scale)))

    # Generate a heat map
    # heat_map = None
    # heat_map = cv2.normalize(sdf_map, heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heat_map = np.uint8(heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

    # Plot true boundary
    edge = cv2.imread(f'{MASK_PATH}{name}.png')
    heat_map = np.maximum(heat_map, edge)

    cv2.imwrite(f'{res_path}{name}.png', heat_map)
    print(f'Heatmap path = {res_path}{name}.png')

    if not show:
        return

    cv2.imshow('SDF Map', heat_map)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    mode = ''
    while mode != 'trained' and mode != 'true':
        print('Choose mode (trained/true):')
        mode = input()

    print('Enter shape name:')
    name = input()

    if mode == 'trained':
        net = True
        path = TRAINED_PATH
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device}!')
        func = SDFNet().to(device)
        if os.path.exists(f'{MODEL_PATH}{name}.pth'):
            func.load_state_dict(torch.load(f'{MODEL_PATH}{name}.pth'))
        else:
            print('Error: No trained data!')
            exit(-1)
    else:
        net = False
        path = TRUE_PATH
        device = 'cpu'
        shape = Polygon()
        shape.load(DATA_PATH, name)
        func = shape.sdf

    print('Plotting results...')
    plot_sdf(func, device, res_path=path, name=name, mask_path=MASK_PATH, is_net=net, show=False)
    print('Done!')
