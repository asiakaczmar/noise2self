import torch
from torchvision.datasets import MNIST

from torchvision import transforms
from torch.utils.data import DataLoader

from scripts.utils import SyntheticNoiseDataset
from models.babyunet import BabyUnet

CHECKPOINTS_PATH = '../checkpoints/'

mnist_test = MNIST('../inferred_data/MNIST', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]), train=False)
noisy_mnist_test = SyntheticNoiseDataset(mnist_test, 'test')
data_loader = DataLoader(noisy_mnist_test, batch_size=256, shuffle=True)

for x in range(0, 200, 10):
    trained_model = BabyUnet()
    trained_model.load_state_dict( CHECKPOINTS_PATH + 'model' + str(x))
    trained_model.eval()

    for i, batch in enumerate(data_loader):
        denoised = trained_model(batch)
        break()
    np.save(denoised.numpy(), '../inferred_data/model' + str(x) + '.npz')

