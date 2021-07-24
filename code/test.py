import numpy as np

f = open('../shapes/polygon_2021-07-24_15-12-35.txt', 'r')
shape = []
line = f.readline()
while line:
    x, y = map(lambda n: np.double(n), line.strip('\n').split(' '))
    shape.append([x, y])
    line = f.readline()
f.close()
s = np.array(shape)

direction = (s[1] - s[0])
d = np.random.uniform(0, 1, size=(2, 1))
boundary_points = s[0] + d * direction
for i in range(1, 10):
    j = 0 if i + 1 >= 10 else i + 1
    direction = (s[j] - s[i])
    d = np.random.uniform(0, 1, size=(2, 1))
    boundary_points = np.concatenate((boundary_points, s[i] + d * direction), axis=0)

print(boundary_points)
noise_1 = np.random.normal(loc=0, scale=np.sqrt(0.05), size=boundary_points.shape)
print(noise_1)
print((boundary_points+noise_1).shape)
