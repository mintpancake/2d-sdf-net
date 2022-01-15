# 2d-sdf-net

This is a neural network approximating the two-dimensional signed distance functions of polygons.
The network structure references DeepSDF: https://github.com/facebookresearch/DeepSDF.

## Intro

* The signed distance function determines the minimum distance of a point to the boundary of a curve. 
* The function is negative inside and positive outside. 
* The zero-contour is the boundary of the curve.

##### Sampled curve

![sampled_vase](https://raw.githubusercontent.com/mintpancake/gallery/main/images/sampled_vase.png)

##### Before truncation

![sdf](https://raw.githubusercontent.com/mintpancake/gallery/main/images/sdf.png)

##### After truncation

![tsdf](https://raw.githubusercontent.com/mintpancake/gallery/main/images/tsdf.png)

### Requirements

* PyTorch
* Tensorboard
* OpenCV

### Get Started

1. `code/drawer.py`
2. `code/sampler.py`
3. `code/trainer.py`
4. `code/renderer.py`
