import torch
import torch.nn as nn
from losses.primitive_loss import PrimitiveLoss
from losses.reconstruction_loss import ReconstructionLoss


class Loss(nn.Module):
	def __init__(self, prim_loss_weight=0):
		super(Loss, self).__init__()

		self.prim_loss_weight = prim_loss_weight
		self.recon_loss = ReconstructionLoss()
		self.primitive_loss = PrimitiveLoss()


	# Compute reconstruction and primitive loss
	def forward(self, target_points, target_distances, refined_distances, csg_model):
		refined_recon_loss = self.recon_loss(target_distances, refined_distances)
		primitive_loss = self.prim_loss_weight * self.primitive_loss(target_points, csg_model)
		total_loss = refined_recon_loss + primitive_loss
		return total_loss


# Test loss
def test():
	batch_size = 2
	num_points = 2
	prim_loss_weight = 0.001
	shape_loss_weight = 0.001
	op_loss_weight = 0.001

	target_distances = torch.rand([batch_size, num_points])
	refined_distances = torch.rand([batch_size, num_points])
	loss = Loss(clamp_dist, prim_loss_weight, shape_loss_weight, op_loss_weight)

	print('Total Loss:')
	print(loss.forward(target_distances, refined_distances, shape_probs, operation_probs))