import torch
import torch.nn as nn
from utilities.csg_model import CSGModel
from losses.reconstruction_loss import ReconstructionLoss


# 
class PrimitiveLoss(nn.Module):
	def __init__(self):
		super(PrimitiveLoss, self).__init__()
		self.recon_loss = ReconstructionLoss()

	def forward(self, refined_distances, primitive_distances, operation_samples):
		primitive_loss = 0

		for primitive_index in range(len(primitive_distances)):
			partial_model_distances = operation_samples[primitive_index]
			recon_loss(refined_distances)
