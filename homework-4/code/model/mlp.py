import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

	def __init__(self, config, input_dim=784, output_dim=10):
		super(MLP, self).__init__()

		HIDDEN_DIM_1 = config.model.mlp.hidden_layer_dim_1
		HIDDEN_DIM_2 = config.model.mlp.hidden_layer_dim_2

		self.fc1 = nn.Linear(input_dim, HIDDEN_DIM_1)
		self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
		self.fc3 = nn.Linear(HIDDEN_DIM_2, output_dim)

	def forward(self, input):
		# input has dimension N, C, H*W
		x = F.relu(self.fc1(input))		# input layer
		x = F.relu(self.fc2(x))			# hidden layer 1
		logits = self.fc3(x)			# hidden layer 2
		#pred = F.softmax(logits, dim=1) # output predictions
		return logits
		