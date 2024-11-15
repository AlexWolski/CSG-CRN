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


	def forward(self, target_sdf, refined_sdf, shape_probs=None, operation_probs=None):
		# Compute reconstruction and regularizer losses
		refined_recon_loss = self.recon_loss(target_sdf, refined_sdf)
		primitive_loss =  self.primitive_loss(refined_sdf)

		return refined_recon_loss + self.prim_loss_weight * primitive_loss


# Test loss
def test():
	batch_size = 2
	num_points = 2
	prim_loss_weight = 0.001

	target_sdf = torch.rand([batch_size, num_points])
	refined_sdf = torch.rand([batch_size, num_points])

	shape_probs = torch.tensor([0.2,0.3,0.5], dtype=float).repeat(batch_size,1)
	operation_probs = torch.tensor([0.9,0.1], dtype=float).repeat(batch_size,1)

	loss = Loss(prim_loss_weight)

	print('Total Loss:')
	print(loss.forward(target_sdf, refined_sdf, shape_probs, operation_probs))