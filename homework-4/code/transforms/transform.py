import torch

class Flatten(object):

	def __init__(self):
		super(Flatten, self).__init__()

	def __call__(self, tensor):
		# Flattens an input image tensor to feed into MLP
		# N, 1, H, W --> N, H*W
		return tensor.view(*(tensor.shape[:0]), -1)