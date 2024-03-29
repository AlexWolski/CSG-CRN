import torch
import torch.nn as nn
from losses.entropy_loss import EntropyLoss
from losses.primitive_loss import PrimitiveLoss
from losses.reconstruction_loss import ReconstructionLoss


class Loss(nn.Module):
	def __init__(self, clamp_dist, prim_loss_weight=0, shape_loss_weight=0, op_loss_weight=0):
		super(Loss, self).__init__()

		self.clamp_dist = clamp_dist
		self.l1_loss = torch.nn.L1Loss(reduction='none')
		self.prim_loss_weight = prim_loss_weight
		self.shape_loss_weight = shape_loss_weight
		self.op_loss_weight = op_loss_weight

		self.recon_loss_1 = ReconstructionLoss(self.clamp_dist)
		self.recon_loss_2 = ReconstructionLoss(self.clamp_dist)
		self.primitive_loss = PrimitiveLoss()
		self.entropy_loss_1 = EntropyLoss()
		self.entropy_loss_2 = EntropyLoss()


	def forward(self, target_sdf, initial_sdf, refined_sdf, shape_probs=None, operation_probs=None):
		# Compute change is loss after refinement
		initial_recon_loss = self.recon_loss_1(target_sdf, initial_sdf)
		refined_recon_loss = self.recon_loss_2(target_sdf, refined_sdf)
		delta_loss = refined_recon_loss - initial_recon_loss

		# Compute weighted regularizer losses
		primitive_loss = self.prim_loss_weight * self.primitive_loss(refined_sdf)

		if shape_probs is not None:
			shape_reg_loss = self.shape_loss_weight * self.entropy_loss_1(shape_probs)
		else:
			shape_reg_loss = 0

		if operation_probs is not None:
			operation_reg_loss = self.op_loss_weight * self.entropy_loss_2(operation_probs)
		else:
			operation_reg_loss = 0
		
		# Combine losses
		total_loss = delta_loss + primitive_loss + shape_reg_loss + operation_reg_loss

		return total_loss


# Test loss
def test():
	batch_size = 2
	num_points = 2
	clamp_dist = 0.1
	prim_loss_weight = 0.001
	shape_loss_weight = 0.001
	op_loss_weight = 0.001

	target_sdf = torch.rand([batch_size, num_points]) * clamp_dist
	initial_sdf = torch.rand([batch_size, num_points]) * clamp_dist
	refined_sdf = torch.rand([batch_size, num_points]) * clamp_dist

	shape_probs = torch.tensor([0.2,0.3,0.5], dtype=float).repeat(batch_size,1)
	operation_probs = torch.tensor([0.9,0.1], dtype=float).repeat(batch_size,1)

	loss = Loss(clamp_dist, prim_loss_weight, shape_loss_weight, op_loss_weight)

	print('Total Loss:')
	print(loss.forward(target_sdf, initial_sdf, refined_sdf, shape_probs, operation_probs))