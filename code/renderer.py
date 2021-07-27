import numpy as np
import cv2
import torch


# Adapted from https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
def plot_sdf(sdf_func, device, img_size=800, filepath='', filename='res', is_net=False, show=False):
    # Sample the 2D domain as a regular grid
    low = -0.5
    high = 0.5
    grid_size = 100

    grid = np.linspace(low, high, grid_size + 1)
    if not is_net:
        sdf_map = [[sdf_func(np.float_([x_, y_]))
                    for y_ in grid] for x_ in grid]
        sdf_map = np.array(sdf_map, dtype=np.float)
    else:
        # Input shape is [1, 2]
        sdf_func.eval()
        with torch.no_grad():
            sdf_map = [[sdf_func(torch.Tensor([[x_, y_]]).to(device)).detach().cpu()
                        for y_ in grid] for x_ in grid]
        sdf_map = torch.Tensor(sdf_map).cpu().numpy()

    sdf_map = sdf_map[:-1, :-1]

    # Scale to canvas size
    scale = int(img_size / grid_size)
    sdf_map = np.kron(sdf_map, np.ones((scale, scale)))

    # Generate a heat map
    heat_map = None
    heat_map = cv2.normalize(sdf_map, heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

    cv2.imwrite(f'{filepath}{filename}.png', heat_map)
    print(f'Saved to \"{filepath}{filename}.png\"!')

    if not show:
        return

    cv2.imshow('SDF Map', heat_map)
    cv2.waitKey()
    cv2.destroyAllWindows()
