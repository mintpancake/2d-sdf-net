import numpy as np


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

    def load(self, path, name):
        vertices = []
        f = open(f'{path}{name}.txt', 'r')
        line = f.readline()
        while line:
            x, y = map(lambda n: np.double(n), line.strip('\n').split(' '))
            vertices.append([x, y])
            line = f.readline()
        f.close()
        self.set_v(np.array(vertices, dtype=np.double))
