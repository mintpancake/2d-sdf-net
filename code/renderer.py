import numpy as np
import cv2
import torch


# Adapted from https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
def plot_sdf(sdf_func, device, filepath='', filename='res.png', is_net=False):
    # Sample the 2D domain as a regular grid
    coordinates_linspace = np.linspace(-4, 4, 100)
    y, x = np.meshgrid(coordinates_linspace, coordinates_linspace)
    if not is_net:
        z = [[sdf_func(np.float_([x_, y_]))
              for y_ in coordinates_linspace]
             for x_ in coordinates_linspace]
    else:
        z = [[sdf_func(torch.Tensor([x_, y_]).to(device)).detach().cpu()
              for y_ in coordinates_linspace]
             for x_ in coordinates_linspace]

    z = np.float_(z)

    z = z[:-1, :-1]

    # TODO: use color to differentiate negative and positive
    # z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    # z = np.abs(z) / z_max * 255
    z = 1 / (1 + np.exp(-20 * z)) * 255

    z = np.uint8(z)

    cv2.imwrite(f'{filepath}{filename}', z)
    print(f'Saved to {filepath}{filename}!')
    cv2.imshow('SDF Map', z)
    cv2.waitKey()
    cv2.destroyAllWindows()
