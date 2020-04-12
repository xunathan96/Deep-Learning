import torch
import torch.nn as nn

from model import SentimentNN
from utils import load_config
from utils.data import dataloader
from utils.metrics import calculate_accuracy, display_model_performance
from utils.state import load_model, load_statistics


def evaluate(config):

	train_iter, val_iter, test_iter, pretrained_embeddings = dataloader(config)

	# load model
	model = load_model(config)

	# load statistics
	loss, accuracy = load_statistics(config)
	display_model_performance(loss, accuracy)

	criterion = nn.CrossEntropyLoss()
	test_loss, test_accuracy = 0, 0
	
	for i, data in enumerate(test_iter):
		text, lengths = data.text 		# text has dimensions (seq_len, batch_size), lengths has dimensions (batch_size)
		label = data.label-1			# label has dimension (batch_size) and range [0,4]

		# forward pass
		logits = model.forward(text, lengths)

		loss = criterion(logits, label)

		test_loss += loss.item()
		test_accuracy += calculate_accuracy(logits, label).item()

	avg_loss = test_loss/len(test_iter)
	avg_accuracy = test_accuracy/len(test_iter)

	print("test loss:    ", avg_loss)
	print("test accuracy:", avg_accuracy)



if __name__ == '__main__':
	config_path = "./config/config.yml"
	config = load_config(config_path)
	evaluate(config)
