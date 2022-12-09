import torch
import torch.nn as nn


# Square the minimum distance between the primitive and the closest sample point
# Loss function borrowed from CSGNet
# https://github.com/kimren227/CSGStumpNet/blob/main/loss.py#L4-L12
class PrimitiveLoss(nn.Module):
	def __init__(self):
		super(PrimitiveLoss, self).__init__()

	def forward(self, sdf_samples):
		# Compute primitive loss of each batch
		(primitive_loss, _) = sdf_samples.min(dim=-1, keepdim=True)
		primitive_loss = torch.square(primitive_loss)

		# Compute average primitive loss of all batches
		primitive_loss = torch.mean(primitive_loss)

		return primitive_loss


# Test loss
if __name__ == "__main__":
	batch_size = 2

	sdf_samples = torch.tensor([0.4,0.6,0.8], dtype=float).repeat(batch_size,1)
	primitive_loss = PrimitiveLoss()

	print('Primitive Loss:')
	print(primitive_loss.forward(sdf_samples))