import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import MLP, CNN
from utils import load_config
from utils.data import load_mnist
from utils.metrics import calculate_accuracy, display_model_performance
from utils.state import load_model, load_statistics


def evaluate(config):

	# define model
	if config.model.name == 'mlp':
		model = MLP(config)
	elif config.model.name == 'cnn':
		model = CNN(config)

	# load model & statistics
	model = load_model(config, model)
	loss, accuracy = load_statistics(config)

	# print performance graphs
	display_model_performance(loss, accuracy)

	# load mnist dataset
	train_loader, test_loader = load_mnist(config)
	test_iter = iter(test_loader)
	images, labels = test_iter.next()

	# evaluate accuracy and loss on test data
	logits = model.forward(images)

	test_loss = nn.CrossEntropyLoss()(logits, labels).detach().numpy()
	test_acc = calculate_accuracy(logits.detach().numpy(), labels)

	print("test loss:      ", test_loss)
	print("test accuracy:  ", test_acc)


if __name__ == '__main__':
	config_path = "./config/config.yml"
	config = load_config(config_path)
	evaluate(config)

