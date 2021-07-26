import torch

a = torch.zeros([20, 3])
list = [0, 0]
b = torch.Tensor(list)
print(a[0, 2])
print(b)
print(len(a))
