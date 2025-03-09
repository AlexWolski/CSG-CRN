import torch
import torch.nn as nn
from losses.reconstruction_loss import ReconstructionLoss


class Loss(nn.Module):
	def __init__(self, loss_metric):
		super(Loss, self).__init__()
		self.recon_loss = ReconstructionLoss(loss_metric)


	# Compute reconstruction and primitive loss
	def forward(self, target_samples, csg_model):
		target_points = target_samples[..., :3]
		target_distances = target_samples[..., 3]

		# Sample CSG model and compute loss
		refined_distances = csg_model.sample_csg(target_points)
		return self.recon_loss(target_distances, refined_distances)


# Test loss
def test():
	batch_size = 2
	num_points = 2
	proximity_loss_weight = 0.001
	shape_loss_weight = 0.001
	op_loss_weight = 0.001

	target_distances = torch.rand([batch_size, num_points])
	refined_distances = torch.rand([batch_size, num_points])
	loss = Loss(clamp_dist, proximity_loss_weight, shape_loss_weight, op_loss_weight)

	print('Total Loss:')
	print(loss.forward(target_distances, refined_distances, shape_probs, operation_probs))