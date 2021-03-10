import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from torch import randn

from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.babyunet import BabyUnet

from mask import Masker

CHECKPOINTS_PATH = '../checkpoints/'

VAR = 2.25

mnist_train = MNIST('../data/MNIST', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]), train=True)

mnist_test = MNIST('../data/MNIST', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]), train=False)


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

noisy_mnist_train = SyntheticNoiseDataset(mnist_train, 'train')
noisy_mnist_test = SyntheticNoiseDataset(mnist_test, 'test')


masker = Masker(width=4, mode='interpolate')

model = BabyUnet()


loss_function = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

data_loader = DataLoader(noisy_mnist_train, batch_size=256, shuffle=True)

for epoch in range(200):
    for i, batch in enumerate(data_loader):
        noisy_images, clean_images = batch

        net_input, mask = masker.mask(noisy_images, i)
        net_output = model(net_input)

        loss = loss_function(net_output * mask, noisy_images * mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        torch.save(model, CHECKPOINTS_PATH + 'model' + str(epoch))
