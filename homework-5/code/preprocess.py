import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

from utils import load_config
from utils.data import dataloader

"""
Not really needed...
please ignore this file
"""

def preprocess(config):
	train_iter, val_iter, test_iter = dataloader(config)

	# print batch information
	batch = next(iter(train_iter)) # for batch in train_iter
	print(batch.text)	# input sequence
	print(batch.label)	# groud truth

	# Attention: batch.label in the range [1,5] not [0,4] !!!



if __name__ == '__main__':
	config_path = "./config/config.yml"
	config = load_config(config_path)
	preprocess(config)