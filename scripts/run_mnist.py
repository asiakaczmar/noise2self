import torch
from torchvision.datasets import MNIST
from torchvision import transforms


from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.babyunet import BabyUnet

from mask import Masker
from scripts.utils import SyntheticNoiseDataset

CHECKPOINTS_PATH = '../checkpoints/'


mnist_train = MNIST('../inferred_data/MNIST', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]), train=True)

noisy_mnist_train = SyntheticNoiseDataset(mnist_train, 'train')



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
        torch.save(model.state_dict(), CHECKPOINTS_PATH + 'model' + str(epoch))
