import torch
import torch.nn as nn


# Compute the reconstruction loss between SDFs sampled from two point clouds
# Loss function borrowed from DeepSDF
# https://github.com/facebookresearch/DeepSDF/blob/main/train_deep_sdf.py#L501
class ReconstructionLoss(nn.Module):
	def __init__(self):
		super(ReconstructionLoss, self).__init__()
		self.l1_loss = torch.nn.L1Loss(reduction='mean')

	# Compute average L1 loss of SDF samples of all batches
	def forward(self, target_sdf, predicted_sdf):
		return self.l1_loss(target_sdf, predicted_sdf)


# Test loss
def test():
	batch_size = 2
	num_points = 2

	target_sdf = torch.rand([batch_size, num_points])
	predicted_sdf = torch.rand([batch_size, num_points])

	reconstruction_loss = ReconstructionLoss()

	print('Reconstruction Loss:')
	print(reconstruction_loss.forward(target_sdf, predicted_sdf))