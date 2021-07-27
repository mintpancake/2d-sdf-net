import numpy as np
import cv2
import torch
from renderer import plot_sdf

SHAPE_PATH = '../shapes/'
TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
SAMPLED_IMAGE_PATH = '../datasets/sampled_images/'
HEATMAP_PATH = '../results/true_heatmaps/'

CANVAS_SIZE = np.array([800, 800])  # Keep two dimensions the same
SHAPE_COLOR = (255, 255, 255)
POINT_COLOR = (127, 127, 127)


# The Geometry and Polygon classes are adapted from
# https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
class Geometry(object):
    EPS = 1e-12

    @staticmethod
    def distance_from_point_to_segment(a, b, p):
        res = min(np.linalg.norm(a - p), np.linalg.norm(b - p))
        if (np.linalg.norm(a - b) > Geometry.EPS
                and np.dot(p - a, b - a) > Geometry.EPS
                and np.dot(p - b, a - b) > Geometry.EPS):
            res = abs(np.cross(p - a, b - a) / np.linalg.norm(b - a))
        return res


class Polygon(object):
    def __init__(self):
        self.v = np.array([])
        # Number of vertices/edges
        self.num = 0

    def set_v(self, v):
        self.v = v
        self.num = len(self.v)

    def sdf(self, p):
        return -self.distance(p) if self.inside(p) else self.distance(p)

    def inside(self, p):
        angle_sum = 0
        for i in range(self.num):
            a = self.v[i]
            b = self.v[(i + 1) % self.num]
            angle_sum += np.arctan2(np.cross(a - p, b - p), np.dot(a - p, b - p))
        return abs(angle_sum) > 1

    def distance(self, p):
        res = Geometry.distance_from_point_to_segment(self.v[-1], self.v[0], p)
        for i in range(len(self.v) - 1):
            res = min(res, Geometry.distance_from_point_to_segment(self.v[i], self.v[i + 1], p))
        return res


class ShapeSampler(object):
    def __init__(self, shape_name, shape_path, train_data_path, val_data_path, sampled_image_path,
                 split_ratio=0.8, show_image=False):
        """
        :param shape_name: "file"
        :param shape_path: "dir/"
        :param train_data_path: "dir/"
        :param val_data_path: "dir/"
        :param sampled_image_path: "dir/"
        :param split_ratio: train / (train + val)
        :param show_image: Launch a windows showing sampled image
        """

        self.shape_name = shape_name
        self.shape_path = shape_path

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.sampled_image_path = sampled_image_path

        self.shape = Polygon()
        self.sampled_data = np.array([])
        self.train_data = np.array([])
        self.val_data = np.array([])

        self.split_ratio = split_ratio
        self.show_image = show_image

    def run(self):
        self.load()
        self.normalize()
        self.sample()
        self.save()

    def load(self):
        vertices = []
        f = open(f'{self.shape_path}{self.shape_name}.txt', 'r')
        line = f.readline()
        while line:
            x, y = map(lambda n: np.double(n), line.strip('\n').split(' '))
            vertices.append([x, y])
            line = f.readline()
        f.close()
        self.shape.set_v(np.array(vertices, dtype=np.double))

    def normalize(self, scope=0.4):
        """
        :param scope: Should be less than 0.5
        """
        if self.shape.num == 0:
            return
        # Calculate the center of gravity
        # and translate the shape
        g = np.mean(self.shape.v, axis=0)
        trans_shape = self.shape.v - g
        # Calculate the farthest away point
        # and scale the shape so that it is bounded by the unit circle
        max_dist = np.max(np.linalg.norm(trans_shape, axis=1))
        scaling_factor = scope / max_dist
        trans_shape *= scaling_factor
        self.shape.v = trans_shape

        # plot_sdf
        plot_sdf(self.shape.sdf, 'cpu', filepath=HEATMAP_PATH, filename=self.shape_name, is_net=False, show=False)

    def sample(self, m=5000, n=2000, var=(0.0025, 0.00025)):
        """
        :param m: number of points sampled on the boundary
                  each boundary point generates 2 samples
        :param n: number of points sampled uniformly in the unit circle
        :param var: two Gaussian variances used to transform boundary points
        """

        if self.shape.num == 0:
            return

        # Do uniform sampling
        # Use polar coordinate
        r = np.random.uniform(0, 0.5, size=(n, 1))
        t = np.random.uniform(0, 2 * np.pi, size=(n, 1))
        # Transform to Cartesian coordinate
        uniform_points = np.concatenate((r * np.cos(t), r * np.sin(t)), axis=1)

        # Do Gaussian sampling
        # Distribute points to each edge weighted by length
        total_length = 0
        edge_length = np.zeros(self.shape.num, dtype=np.double)
        for i in range(self.shape.num):
            length = np.linalg.norm(self.shape.v[(i + 1) % self.shape.num] - self.shape.v[i])
            edge_length[i] = length
            total_length += length
        edge_portion = edge_length / total_length
        edge_portion *= m
        edge_num = np.around(edge_portion).astype(int)

        # Do sampling on edges
        direction = (self.shape.v[1] - self.shape.v[0])
        d = np.random.uniform(0, 1, size=(edge_num[0], 1))
        boundary_points = self.shape.v[0] + d * direction
        for i in range(1, self.shape.num):
            direction = (self.shape.v[(i + 1) % self.shape.num] - self.shape.v[i])
            d = np.random.uniform(0, 1, size=(edge_num[i], 1))
            boundary_points = np.concatenate((boundary_points, self.shape.v[i] + d * direction), axis=0)

        # Perturbing boundary points
        noise_1 = np.random.normal(loc=0, scale=np.sqrt(var[0]), size=boundary_points.shape)
        noise_2 = np.random.normal(loc=0, scale=np.sqrt(var[1]), size=boundary_points.shape)
        gaussian_points = np.concatenate((boundary_points + noise_1, boundary_points + noise_2), axis=0)

        # Merge uniform and Gaussian points
        sampled_points = np.concatenate((uniform_points, gaussian_points), axis=0)
        self.sampled_data = self.calculate_sdf(sampled_points)

        # Split sampled data into train dataset and val dataset
        train_size = int(len(self.sampled_data) * self.split_ratio)
        choice = np.random.choice(range(self.sampled_data.shape[0]), size=(train_size,), replace=False)
        ind = np.zeros(self.sampled_data.shape[0], dtype=bool)
        ind[choice] = True
        rest = ~ind
        self.train_data = self.sampled_data[ind]
        self.val_data = self.sampled_data[rest]

    def calculate_sdf(self, points):
        if self.shape.num == 0:
            return

        # Add a third column for storing sdf
        data = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)
        data[:, 2] = np.apply_along_axis(self.shape.sdf, 1, data[:, :2])
        return data

    def save(self):
        if self.shape.num == 0:
            return

        save_name = self.shape_name

        # Save data to .txt
        f = open(f'{self.train_data_path}{save_name}.txt', 'w')
        for datum in self.train_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]}\n')
        f.close()
        f = open(f'{self.val_data_path}{save_name}.txt', 'w')
        for datum in self.val_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]}\n')
        f.close()

        # Generate a sampled image
        canvas = np.zeros(CANVAS_SIZE, np.uint8)
        # Draw polygon
        scaled_v = np.around(self.shape.v * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int)
        cv2.fillPoly(canvas, scaled_v[np.newaxis, :, :], SHAPE_COLOR)
        # Draw points
        for i, datum in enumerate(self.sampled_data):
            point = np.around(datum[:2] * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int)
            cv2.circle(canvas, point, 1, POINT_COLOR, -1)
            if i % 50 == 0:
                radius = np.abs(np.around(datum[2] * CANVAS_SIZE[0]).astype(int))
                cv2.circle(canvas, point, radius, POINT_COLOR)

        # Store and show
        cv2.imwrite(f'{self.sampled_image_path}{save_name}.png', canvas)

        if not self.show_image:
            return

        cv2.imshow('Sampled Image', canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Enter shape name:')
    shape_name = input()
    sampler = ShapeSampler(shape_name, SHAPE_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, SAMPLED_IMAGE_PATH, show_image=False)
    print('Sampling...')
    sampler.run()
    print('Done!')
