import torch
from torch.utils.data import Dataset, TensorDataset
from torch import randn
import numpy as np

VAR = 1.5

def add_noise(img):
    return img + randn(img.size()) * VAR


def dataset_from_numpy(data_location):
    array = np.load(data_location)
    x = torch.Tensor(array)
    return TensorDataset(x)


class SyntheticNoiseDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.mode = mode
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        return add_noise(img), img

if __name__ == '__main__':
    d = dataset_from_numpy('/Users/siak/PhD/scopem/dGAN_rework/noisy_train.npy')
    print(d)