import torch
import torch.nn as nn
from losses.entropy_loss import EntropyLoss
from losses.primitive_loss import PrimitiveLoss
from losses.reconstruction_loss import ReconstructionLoss


class Loss(nn.Module):
	def __init__(self, clamp_dist, primitive_weight, shape_weight, operation_weight):
		super(Loss, self).__init__()

		self.clamp_dist = clamp_dist
		self.l1_loss = torch.nn.L1Loss(reduction='none')
		self.primitive_weight = primitive_weight
		self.shape_weight = shape_weight
		self.operation_weight = operation_weight

		self.recon_loss_1 = ReconstructionLoss(self.clamp_dist)
		self.recon_loss_2 = ReconstructionLoss(self.clamp_dist)
		self.primitive_loss = PrimitiveLoss()
		self.entropy_loss_1 = EntropyLoss()
		self.entropy_loss_2 = EntropyLoss()


	def forward(self, target_sdf, initial_sdf, refined_sdf, shape_probs, operation_probs):
		# Compute change is loss after refinement
		initial_recon_loss = self.recon_loss_1(target_sdf, initial_sdf)
		refined_recon_loss = self.recon_loss_2(target_sdf, refined_sdf)
		delta_loss = initial_recon_loss - refined_recon_loss

		# Compute weighted regularizer losses
		primitive_loss = self.primitive_weight * self.primitive_loss(refined_sdf)
		shape_reg_loss = self.shape_weight * self.entropy_loss_1(shape_probs)
		operation_reg_loss = self.operation_weight * self.entropy_loss_2(operation_probs)

		# Combine losses
		total_loss = delta_loss + primitive_loss + shape_reg_loss + operation_reg_loss

		return total_loss


# Test loss
def test():
	batch_size = 2
	num_points = 2
	clamp_dist = 0.1
	primitive_weight = 0.001
	shape_weight = 0.001
	operation_weight = 0.001

	target_sdf = torch.rand([batch_size, num_points]) * clamp_dist
	initial_sdf = torch.rand([batch_size, num_points]) * clamp_dist
	refined_sdf = torch.rand([batch_size, num_points]) * clamp_dist

	shape_probs = torch.tensor([0.2,0.3,0.5], dtype=float).repeat(batch_size,1)
	operation_probs = torch.tensor([0.9,0.1], dtype=float).repeat(batch_size,1)

	loss = Loss(clamp_dist, primitive_weight, shape_weight, operation_weight)

	print('Total Loss:')
	print(loss.forward(target_sdf, initial_sdf, refined_sdf, shape_probs, operation_probs))