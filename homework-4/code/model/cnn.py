import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

	def __init__(self, config, input_channels=1, output_dim=10):
		super(CNN, self).__init__()

		N_CHANNELS_1 = config.model.cnn.n_channels_1
		N_CHANNELS_2 = config.model.cnn.n_channels_2
		HIDDEN_DIM = config.model.cnn.hidden_layer_dim
		KERNEL_SIZE = config.model.cnn.kernel_size
		PAD = KERNEL_SIZE//2
		FLAT_DIM = N_CHANNELS_2*7*7

		self.conv1 = nn.Conv2d(input_channels, N_CHANNELS_1, KERNEL_SIZE, padding=PAD)
		self.conv2 = nn.Conv2d(N_CHANNELS_1, N_CHANNELS_2, KERNEL_SIZE, padding=PAD)
		self.fc1 = nn.Linear(FLAT_DIM, HIDDEN_DIM)
		self.fc2 = nn.Linear(HIDDEN_DIM, output_dim)

	# FORWARD PASS
	# input has dimensions (N, C, H, W)
	# N - batch size
	# C - number of channels
	# H, W - height and width of image
	def forward(self, input):

		# conv - relu - maxpool layers
		x = F.max_pool2d(F.relu(self.conv1(input)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

		# flatten layer
		x = x.view(-1, self.num_flat_features(x))

		# fc layers
		x = F.relu(self.fc1(x))
		logits = self.fc2(x)

		return logits


	# Helper function that gets the dimensions of the flattened feature map
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

