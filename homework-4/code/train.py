import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from model import MLP, CNN
from utils import load_config
from utils.data import load_mnist
from utils.metrics import calculate_accuracy
from utils.state import save_model, save_statistics

def train(config):

	# load mnist dataset
	train_loader, test_loader = load_mnist(config)

	# define model
	if config.model.name == 'mlp':
		model = MLP(config)
	elif config.model.name == 'cnn':
		model = CNN(config)
	
	# define loss criteria & optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(
		model.parameters(), 
		lr=config.optimizer.params.lr, 
		momentum=config.optimizer.params.momentum,
		weight_decay=config.optimizer.params.regularization
	)


	BATCH_SIZE = config.data.mnist.batch_size
	MAX_EPOCH = config.optimizer.epochs
	BATCHES_PER_EPOCH = len(train_loader)

	loss_batch = []
	acc_batch = []
	running_loss, running_acc = 0, 0

	for epoch in range(MAX_EPOCH):
		batch_loss, batch_acc = 0, 0
		
		for i, data in enumerate(train_loader, 0):
			inputs, labels = data

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward pass
			logits = model.forward(inputs)

			# calculate softmax cross-entropy loss
			loss = criterion(logits, labels)

			# backward pass & gradient descent
			loss.backward()
			optimizer.step()

			# fetch statistics for the current batch
			acc = calculate_accuracy(logits.detach().numpy(), labels)
			running_acc += acc
			running_loss += loss.item()
			batch_loss += loss.item()
			batch_acc += acc

			# print running statistics every 50 batches
			if i % 50 == 49:
				print('Epoch [{}]/[{}]\t Batch [{}]/[{}]\t loss: [{:.3f}]\t accuracy: [{:.4f}]'
					.format(epoch+1, MAX_EPOCH, i+1, BATCHES_PER_EPOCH, running_loss/50, running_acc/50))
				running_loss, running_acc = 0, 0

		# save batch statistics
		loss_batch.append(batch_loss/BATCH_SIZE)
		acc_batch.append(batch_acc/BATCH_SIZE)


	# save model and loss & accuracy curves
	save_model(model, config)
	save_statistics(loss_batch, acc_batch, config)


if __name__ == '__main__':
	config_path = "./config/config.yml"
	config = load_config(config_path)
	train(config)




