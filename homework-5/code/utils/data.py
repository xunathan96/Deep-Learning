import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText


def dataloader(config):

	ROOT = config.data.dir
	BATCH_SIZE = config.data.batch_size
	EMBEDDING_DIM = config.data.embedding_size

	# set up fields
	TEXT = data.Field(include_lengths=True)					# use include_lengths=True to cause batch.text to be a tuple (sentence, sentence_length)
	LABEL = data.Field(sequential=False,dtype=torch.long)

	# make splits for data
	# DO NOT MODIFY: fine_grained=True, train_subtrees=False
	train, val, test = datasets.SST.splits(
		TEXT, LABEL, 
		root=ROOT+'.data', 
		fine_grained=True, 
		train_subtrees=False
	)

	# build the vocabulary
	# you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
	TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache=ROOT))
	LABEL.build_vocab(train)
	pretrained_embeddings = TEXT.vocab.vectors

	# zero weights of unknown and padding embeddings
	UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
	PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
	pretrained_embeddings[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
	pretrained_embeddings[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

	# make iterator for splits
	train_iter, val_iter, test_iter = data.BucketIterator.splits(
		(train, val, test), 
		batch_size=BATCH_SIZE,
		sort_within_batch=True 	# for packed padded sequences all of the tensors within a batch need to be sorted by their lengths
	)

	return train_iter, val_iter, test_iter, pretrained_embeddings


	"""
	# print batch information
	batch = next(iter(train_iter)) # for batch in train_iter
	print(batch.text) # input sequence
	print(batch.label) # groud truth

	# Attention: batch.label in the range [1,5] not [0,4] !!!
	"""





