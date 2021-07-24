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
        self.shape = np.array(shape, dtype=np.double)

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

        # Do Gaussian sampling
        # Distribute points to each edge weighted by length
        total_length = 0
        edge_length = np.zeros(self.num, dtype=np.double)
        for i in range(self.num):
            j = 0 if i + 1 >= self.num else i + 1
            length = np.linalg.norm(self.shape[j] - self.shape[i])
            edge_length[i] = length
            total_length += length
        edge_portion = edge_length / total_length
        edge_portion *= m
        edge_num = np.around(edge_portion).astype(int)

        # Do sampling on edges
        direction = (self.shape[1] - self.shape[0])
        d = np.random.uniform(0, 1, size=(edge_num[0], 1))
        boundary_points = self.shape[0] + d * direction
        for i in range(1, self.num):
            j = 0 if i + 1 >= self.num else i + 1
            direction = (self.shape[j] - self.shape[i])
            d = np.random.uniform(0, 1, size=(edge_num[i], 1))
            boundary_points = np.concatenate((boundary_points, self.shape[i] + d * direction), axis=0)

        # Perturbing boundary points
        noise_1 = np.random.normal(loc=0, scale=np.sqrt(var[0]), size=boundary_points.shape)
        noise_2 = np.random.normal(loc=0, scale=np.sqrt(var[1]), size=boundary_points.shape)
        near_bound_points = np.concatenate((boundary_points + noise_1, boundary_points + noise_2), axis=0)
        # Add a third column for sdf
        gaussian_points = np.concatenate((near_bound_points, np.zeros((near_bound_points.shape[0], 1))), axis=1)

        # Merge uniform and Gaussian points
        sampled_points = np.concatenate((uniform_points, gaussian_points), axis=0)
        self.sampled_data = self.calculate_sdf(sampled_points)

    def calculate_sdf(self, points):
        data = np.array([])
        return data

    def save(self):
        pass


if __name__ == '__main__':
    sampler = ShapeSampler(SHAPE_PATH, SHAPE_IMAGE_PATH)
    sampled_data = sampler.run()
