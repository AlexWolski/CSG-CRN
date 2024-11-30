import torch
import torch.nn as nn
from losses.proximity_loss import ProximityLoss
from losses.reconstruction_loss import ReconstructionLoss


class Loss(nn.Module):
	def __init__(self, proximity_loss_weight=0):
		super(Loss, self).__init__()

		self.proximity_loss_weight = proximity_loss_weight
		self.recon_loss = ReconstructionLoss()
		self.proximity_loss = ProximityLoss()


	# Compute reconstruction and primitive loss
	def forward(self, target_samples, init_recon_samples, csg_model):
		target_points = target_samples[..., :3]
		target_distances = target_samples[..., 3]
		init_recon_points = init_recon_samples[..., :3] if init_recon_samples is not None else None
		init_recon_distances = init_recon_samples[..., 3] if init_recon_samples is not None else None

		# Sample CSG model
		primitive_distances = []
		refined_distances = csg_model.sample_csg(target_points, initial_distances=init_recon_distances, out_primitive_samples=primitive_distances)

		# Compute loss
		refined_recon_loss = self.recon_loss(target_distances, refined_distances)
		proximity_loss = self.proximity_loss_weight * self.proximity_loss(primitive_distances)
		total_loss = refined_recon_loss + proximity_loss
		return total_loss


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