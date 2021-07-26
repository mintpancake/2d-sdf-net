import torch
from torch.utils.data import Dataset


class SDFData(Dataset):
    def __init__(self, file):
        """
        :param file: Should be in the format of "dir/file.txt"
        """
        self.file = file
        self.data = self.load()

    def __getitem__(self, item):
        xy = self.data[item, :2]
        sdf = self.data[item, 2]
        return xy, sdf

    def __len__(self):
        return len(self.data)

    def load(self):
        data = []
        f = open(self.file, 'r')
        line = f.readline()
        while line:
            x, y, sdf = map(lambda n: float(n), line.strip('\n').split(' '))
            data.append([x, y, sdf])
            line = f.readline()
        f.close()
        return torch.Tensor(data)
