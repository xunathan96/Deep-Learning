import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data

from model import SentimentNN
from utils import load_config
from utils.data import dataloader
from utils.metrics import calculate_accuracy
from utils.state import save_model, save_statistics

def train(config):

	train_iter, val_iter, test_iter, pretrained_embeddings = dataloader(config)
	vocab_size, embedding_dim = pretrained_embeddings.shape

	# define model
	model = SentimentNN(config, pretrained_embeddings)

	# define loss criteria & optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(
		model.parameters(), 
		lr=config.optimizer.params.lr, 
		weight_decay=config.optimizer.params.regularization
	)

	N_EPOCHS = config.optimizer.epochs
	running_loss, running_accuracy = 0, 0
	loss_batch = []
	acc_batch = []

	for epoch in range(N_EPOCHS):
		batch_loss, batch_acc = 0, 0
		for i, data in enumerate(train_iter):
			text, lengths = data.text 		# text has dimensions (seq_len, batch_size), lengths has dimensions (batch_size)
			label = data.label-1			# label has dimension (batch_size) and range [0,4]

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward pass
			logits = model.forward(text, lengths)

			# calculate softmax cross-entropy loss
			loss = criterion(logits, label)

			# backward pass & gradient descent
			loss.backward()
			optimizer.step()

			# get statistics for the current batch
			accuracy = calculate_accuracy(logits, label).item()
			running_loss += loss.item()
			running_accuracy += accuracy
			batch_loss += loss.item()
			batch_acc += accuracy

			if i%5 == 0:
				print('Epoch: {}\t Batch: {}\t loss: [{:.3f}]\t accuracy: [{:.4f}]'
					.format(epoch+1, i, running_loss/5, running_accuracy/5)
				)
				running_loss, running_accuracy = 0, 0

		# save batch statistics
		loss_batch.append(batch_loss/len(train_iter))
		acc_batch.append(batch_acc/len(train_iter))

	#save model and loss & accuracy curves
	save_model(model, config)
	save_statistics(loss_batch, acc_batch, config)



if __name__ == '__main__':
	config_path = "./config/config.yml"
	config = load_config(config_path)
	train(config)

