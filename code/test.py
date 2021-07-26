from renderer import *
from sampler import *

sampler = ShapeSampler('star', SHAPE_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, SAMPLED_IMAGE_PATH)
sampler.load()
sampler.normalize()
plot_sdf(sampler.shape.sdf, 'cpu', CANVAS_SIZE[0], filepath='../results/', filename='test_star.png', is_net=False)
