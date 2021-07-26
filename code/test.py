from renderer import *
from sampler import *

sampler = ShapeSampler('square', SHAPE_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, SAMPLED_IMAGE_PATH)
sampler.load()
sampler.normalize()
plot_sdf(sampler.shape.sdf, 'cpu', filepath='../results/', filename='test_square.png', is_net=False)
