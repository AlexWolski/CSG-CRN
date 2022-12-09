import torch
import torch.nn as nn


# Constant to add to probability distribution to prevent computing log of zero
EPSILON = 1e-7


# Compute the entropy of a multinomial distribution
# Input Shape: BxPxF
# Where B = Batch Size, P = Number of Primitives, F = Number of Features
# Implementation borrowed from SciPy module
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
# S = -sum(pk * log(pk), axis=axis)
class EntropyLoss(nn.Module):
	def __init__(self):
		super(EntropyLoss, self).__init__()

	def forward(self, prob_distribution):
		prob_distribution = torch.add(prob_distribution, EPSILON)
		categorical_entropy = -prob_distribution * torch.log(prob_distribution)
		total_entropy = torch.sum(categorical_entropy, axis=-1)

		return total_entropy


# Test loss
def test():
	batch_size = 2

	shape_dist = torch.tensor([0.2,0.3,0.5], dtype=float).repeat(batch_size,1)
	operation_dist = torch.tensor([0.9,0.1], dtype=float).repeat(batch_size,1)

	entropy_loss = EntropyLoss()

	print('Shape Entropy:')
	print(entropy_loss.forward(shape_dist))

	print('Operation Entropy:')
	print(entropy_loss.forward(operation_dist))