# 2d-sdf-net

This is a neural network approximating the two-dimensional signed distance functions of polygons.
The network structure references DeepSDF: https://github.com/facebookresearch/DeepSDF.

## Intro

* The signed distance function determines the minimum distance of a point to the boundary of a curve. 
* The function is negative inside and positive outside. 
* The zero-contour is the boundary of the curve.

##### Sampled curve

<div align=center><img width="400" height="400" src="https://raw.githubusercontent.com/mintpancake/gallery/main/images/sampled_vase.png"/></div>

##### Before truncation

<div align=center><img width="400" height="400" src="https://raw.githubusercontent.com/mintpancake/gallery/main/images/sdf.png"/></div>

##### After truncation

<div align=center><img width="400" height="400" src="https://raw.githubusercontent.com/mintpancake/gallery/main/images/tsdf.png"/></div>

### Requirements

* PyTorch
* Tensorboard
* OpenCV

### Get Started

1. `code/drawer.py`
2. `code/sampler.py`
3. `code/trainer.py`
4. `code/renderer.py`
