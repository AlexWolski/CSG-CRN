import torch
import torch.nn as nn
from utilities.csg_model import CSGModel


# Square the minimum distance between the primitive and the closest sample point
# Loss function inspired by CSGNet
# https://github.com/kimren227/CSGStumpNet/blob/main/loss.py#L4-L12
class PrimitiveLoss(nn.Module):
	def __init__(self):
		super(PrimitiveLoss, self).__init__()

	def forward(self, target_points, primitive_distances):
		primitive_distances = torch.stack(primitive_distances)
		min_distances = torch.square(primitive_distances)
		min_distances = torch.amin(min_distances, dim=-1, keepdim=True)
		return torch.mean(min_distances)


# Test loss
def test():
	batch_size = 2

	sdf_samples = torch.tensor([0.4,0.6,0.8], dtype=float).repeat(batch_size,1)
	primitive_loss = PrimitiveLoss()

	print('Primitive Loss:')
	print(primitive_loss.forward(sdf_samples))