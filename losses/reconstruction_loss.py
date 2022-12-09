import torch
import torch.nn as nn


# Compute the reconstruction loss between SDFs sampled from two point clouds
# Loss function borrowed from DeepSDF
# https://github.com/facebookresearch/DeepSDF/blob/main/train_deep_sdf.py#L501
class ReconstructionLoss(nn.Module):
	def __init__(self, clamp_dist):
		super(ReconstructionLoss, self).__init__()
		self.clamp_dist = clamp_dist
		self.l1_loss = torch.nn.L1Loss(reduction='none')

	def forward(self, target_sdf, predicted_sdf):
		num_samples = target_sdf.size(dim=-1)

		# Clamp SDF values to focus loss on nearby points
		target_sdf_clamped = torch.clamp(target_sdf, -self.clamp_dist, self.clamp_dist)
		predicted_sdf_clamped = torch.clamp(predicted_sdf, -self.clamp_dist, self.clamp_dist)

		# Compute average L1 loss of SDF samples for each batch
		recon_loss = self.l1_loss(target_sdf_clamped, predicted_sdf_clamped)
		# Compute average loss of all batches
		recon_loss = torch.mean(recon_loss)

		return recon_loss


# Test loss
def test():
	batch_size = 2
	num_points = 2
	clamp_dist = 0.1

	target_sdf = torch.rand([batch_size, num_points])
	predicted_sdf = torch.rand([batch_size, num_points])

	reconstruction_loss = ReconstructionLoss(clamp_dist)

	print('Reconstruction Loss:')
	print(reconstruction_loss.forward(target_sdf, predicted_sdf))