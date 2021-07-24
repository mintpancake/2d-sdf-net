import numpy as np

SHAPE_PATH = '../shapes/'
SHAPE_IMAGE_PATH = '../shapes/shape_images/'
SAMPLED_DATA_PATH = '../datasets/'
SAMPLED_IMAGE_PATH = '../datasets/sampled_images/'


class ShapeSampler(object):
    def __init__(self, shape_path, shape_name):
        self.shape_path = shape_path
        self.shape_name = shape_name
        self.shape = np.array([])
        # Number of vertices/edges
        self.num = 0
        self.sampled_data = np.array([])

    def run(self):
        self.load()
        self.normalize()
        self.sample()
        self.save()
        return self.sampled_data

    def load(self):
        shape = []
        f = open(f'{self.shape_path}{self.shape_name}.txt', 'r')
        line = f.readline()
        while line:
            x, y = map(lambda n: np.double(n), line.strip('\n').split(' '))
            shape.append([x, y])
            self.num += 1
            line = f.readline()
        f.close()
        self.shape = np.array(shape)

    def normalize(self, margin=0.9):
        if self.num == 0:
            return
        # Calculate the center of gravity
        # and translate the shape
        g = np.mean(self.shape, axis=0)
        trans_shape = self.shape - g
        # Calculate the farthest away point
        # and scale the shape so that it is bounded by the unit circle
        max_dist = np.max(np.linalg.norm(trans_shape, axis=1))
        scaling_factor = margin / max_dist
        trans_shape *= scaling_factor
        self.shape = trans_shape

    def sample(self, m=4000, n=1000, var=(0.0025, 0.00025)):
        """
        m: number of points sampled on the boundary
           each boundary point generates 2 samples
        n: number of points sampled uniformly in the unit circle
        var: two Gaussian variances used to transform boundary points
        """

        if self.num == 0:
            return

        # Do uniform sampling
        # Use polar coordinate
        r = np.random.uniform(0, 1, size=(n, 1))
        t = np.random.uniform(0, 2 * np.pi, size=(n, 1))
        # Transform to Cartesian coordinate
        uniform_points = np.concatenate((r * np.cos(t), r * np.sin(t), np.zeros((n, 1))), axis=1)

        # Do boundary sampling
        # Distribute points to each edge weighted by length
        total_length = 0
        edge_portion = np.zeros(self.num, dtype=np.double)
        for i in range(self.num):
            j = i + 1
            if j >= self.num:
                j = 0
            length = np.linalg.norm(self.shape[i] - self.shape[j])
            edge_portion[i] = length
            total_length += length
        edge_portion /= total_length
        edge_portion *= m
        num_per_edge = np.around(edge_portion).astype(int)

    def calculate_sdf(self):
        pass

    def save(self):
        pass


if __name__ == '__main__':
    sampler = ShapeSampler(SHAPE_PATH, SHAPE_IMAGE_PATH)
    sampled_data = sampler.run()
