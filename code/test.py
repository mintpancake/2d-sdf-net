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
print(s)
a = np.linalg.norm(s[0] - s[1])
print(a)
a+=0.5
b=np.around(a).astype(int)
print(b)