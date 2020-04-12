import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentimentNN(nn.Module):

	def __init__(self, config, pretrained_embeddings):
		super(SentimentNN, self).__init__()

		self.config = config
		HIDDEN_DIM = config.model.hidden_dim
		N_LAYERS = config.model.n_layers
		DROPOUT = config.model.dropout
		BIDIRECTIONAL = config.model.bidirectional
		EMBEDDING_DIM = pretrained_embeddings.shape[1]
		DROPOUT = self.config.model.dropout

		self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)

		self.lstm = nn.LSTM(
			input_size = EMBEDDING_DIM, 
			hidden_size = HIDDEN_DIM,
			num_layers = N_LAYERS,
			dropout = DROPOUT,
			bidirectional = BIDIRECTIONAL
		)

		self.dropout = nn.Dropout(p=DROPOUT)

		if BIDIRECTIONAL:
			HIDDEN_DIM *= 2
		self.fc = nn.Linear(HIDDEN_DIM, 5)

	"""
	input has dimensions (seq_length, batch_size)
	we use the lookup table/embedding layer to look up the embeddings for each word id, 
	then pass this batched sequential data through the LSTM.
	The output of the LSTM is passed through a linear layer to get the logits.
	"""
	def forward(self, input, lengths):

		HIDDEN_DIM = self.config.model.hidden_dim
		N_LAYERS = self.config.model.n_layers
		N_DIRECTIONS = 1 if self.config.model.bidirectional==False else 2

		# embeddings has shape (seq_len, batch_size, emb_dim)
		embeddings = self.embedding(input)

		# pack sequence
		# this will allow the RNN to only process the non-padded elements of our sequence.
		packed_embeddings = pack_padded_sequence(embeddings, lengths)

		# forward pass through the LSTM for each data in sequence
		# without packed padded sequences, hidden and cell are tensors from the last element in the sequence, 
		# which will most probably be a pad token, however when using packed padded sequences they are both 
		# from the last non-padded element in the sequence
		packed_hidden, (last_hidden, last_cell) = self.lstm(packed_embeddings)
		#   h has dim: seq_len, batch, num_directions * hidden_size
		# h_n has dim: num_layers * num_directions, batch, hidden_size
		# c_n has dim: num_layers * num_directions, batch, hidden_size

		# unpack sequence
		hidden, lengths = pad_packed_sequence(packed_hidden)

		# dropout on lstm output
		last_hidden = self.dropout(last_hidden)

		# last_hidden reshaped to dim: (num_layers, num_directions, batch, hidden_size)
		last_hidden = last_hidden.view(N_LAYERS, N_DIRECTIONS, *(last_hidden.shape[1:]))

		# output of bi-directional lstm is taken from a concatentation of forward and backward of last layer
		if N_DIRECTIONS==2:
			forward_dir = last_hidden[N_LAYERS-1, 0]
			backward_dir = last_hidden[N_LAYERS-1, 1]
			x = torch.cat((forward_dir, backward_dir), dim=1)
		
		# output of lstm is taken from last layer, forward direction
		else:
			x = last_hidden[N_LAYERS-1, 0]


		logits = self.fc(x)

		return logits



