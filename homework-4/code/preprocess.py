import torchvision

from utils import load_config
from torch.utils.data import DataLoader

"""
No need to preprocess any data here...
please ignore
"""


def main(config):
	
	MNIST_PATH = config.data.mnist.train.path
	BATCH_SIZE = config.data.mnist.batch_size

	train_data = torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True)
	train_loader = DataLoader(
		dataset = train_data, 
		batch_size = BATCH_SIZE,
		shuffle = True,
		num_workers = 2
	)

	test_data = torchvision.datasets.MNIST(MNIST_PATH, train=False, download=True)
	test_loader = DataLoader(
		dataset = train_data, 
		batch_size = BATCH_SIZE,
		shuffle = True,
		num_workers = 2
	)

	

if __name__ == '__main__':
	config_path = "./config/config.yml"
	config = load_config(config_path)
	main(config)