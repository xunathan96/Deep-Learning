import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from transforms import Flatten

def load_mnist(config):
	MNIST_PATH = config.data.mnist.train.path
	BATCH_SIZE = config.data.mnist.batch_size
	MODEL = config.model.name

	transforms_list = [transforms.ToTensor()] + ([Flatten()] if MODEL == 'mlp' else [])
	transform = transforms.Compose(transforms_list)

	train_data = torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True, transform=transform)
	train_loader = DataLoader(
		dataset = train_data, 
		batch_size = BATCH_SIZE,
		shuffle = True,
		num_workers = 2
	)

	test_data = torchvision.datasets.MNIST(MNIST_PATH, train=False, download=True, transform=transform)
	test_loader = DataLoader(
		dataset = test_data, 
		batch_size = len(test_data),
		shuffle = True,
		num_workers = 2
	)

	return train_loader, test_loader
