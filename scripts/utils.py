from torch.utils.data import Dataset
from torch import randn


VAR = 2.25

def add_noise(img):
    return img + randn(img.size()) * VAR


class SyntheticNoiseDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.mode = mode
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        return add_noise(img), img