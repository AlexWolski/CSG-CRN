import torch
import torch.nn as nn


# Compute the reconstruction loss between SDFs sampled from two point clouds
# Loss function borrowed from DeepSDF
# https://github.com/facebookresearch/DeepSDF/blob/main/train_deep_sdf.py#L501
class ReconstructionLoss(nn.Module):
	def __init__(self):
		super(ReconstructionLoss, self).__init__()
		self.l2_loss = torch.nn.MSELoss(reduction='none')

	def forward(self, target_sdf, predicted_sdf):
		# Compute average L2 loss of SDF samples for each batch
		recon_loss = self.l2_loss(target_sdf, predicted_sdf)
		# Compute average loss of all batches
		recon_loss = torch.mean(recon_loss)

		return recon_loss


# Test loss
def test():
	batch_size = 2
	num_points = 2

	target_sdf = torch.rand([batch_size, num_points])
	predicted_sdf = torch.rand([batch_size, num_points])

	reconstruction_loss = ReconstructionLoss()

	print('Reconstruction Loss:')
	print(reconstruction_loss.forward(target_sdf, predicted_sdf))