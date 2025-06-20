import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss


# Compute the reconstruction loss between SDFs sampled from two point clouds
# Loss function borrowed from DeepSDF
# https://github.com/facebookresearch/DeepSDF/blob/main/train_deep_sdf.py#L501
class ReconstructionLoss(nn.Module):
	L1_LOSS_FUNC = "L1"
	MSE_LOSS_FUNC = "MSE"
	LOG_LOSS_FUNC = "LOG"
	OCC_LOSS_FUNC = "OCC"


	def __init__(self, loss_metric, clamp_dist=None):
		super(ReconstructionLoss, self).__init__()

		self.clamp_dist = clamp_dist

		# Select loss function
		match loss_metric:
			case self.L1_LOSS_FUNC:
				self.loss_func = torch.nn.L1Loss(reduction='none')
			case self.MSE_LOSS_FUNC:
				self.loss_func = torch.nn.MSELoss(reduction='none')
			case self.LOG_LOSS_FUNC:
				self.loss_func = LogLoss(reduction='none')
			case self.OCC_LOSS_FUNC:
				self.loss_func = OccLoss(reduction='none')
			case None:
				raise Exception("A loss function must be provided")
			case _:
				raise Exception(f"Invalid loss metric: {loss_metric}")


	# Compute average loss of SDF samples of all batches
	def forward(self, target_sdf, predicted_sdf):
		# Clamp the predicted SDF values to have a maximum difference of clamp_dist*2
		if self.clamp_dist != None:
			clamped_target_sdf = torch.clamp(target_sdf, min=target_sdf-self.clamp_dist, max=predicted_sdf+self.clamp_dist)
			clamped_predicted_sdf = torch.clamp(predicted_sdf, min=predicted_sdf-self.clamp_dist, max=predicted_sdf+self.clamp_dist)
		else:
			clamped_target_sdf = target_sdf
			clamped_predicted_sdf = predicted_sdf

		# Compute the loss
		loss_tensor = self.loss_func(clamped_target_sdf, clamped_predicted_sdf)
		return torch.mean(loss_tensor)


# Log loss function roughly intersecting (0,0)
class LogLoss(_Loss):
	def __init__(self, reduction: str = 'mean'):
		super(LogLoss, self).__init__(None, None, reduction)


	def forward(self, input: Tensor, target: Tensor) -> Tensor:
		return LogLoss.log_loss(input, target, self.reduction)


	def log_loss(target_sdf, predicted_sdf, reduction='mean'):
		X_OFFSET = 0.01
		Y_OFFSET = 1
		MULTIPLE = 5

		expanded_target, expanded_predicted = torch.broadcast_tensors(target_sdf, predicted_sdf)
		absolute_error = torch.abs(expanded_target - predicted_sdf)
		log_loss = torch.log(absolute_error + X_OFFSET) / MULTIPLE + Y_OFFSET

		match reduction:
			case 'none':
				return log_loss
			case 'sum':
				return torch.sum(log_loss)
			case 'mean':
				return torch.mean(log_loss)


class OccLoss(_Loss):
	def __init__(self, reduction: str = 'mean'):
		super(OccLoss, self).__init__(None, None, reduction)


	def forward(self, input: Tensor, target: Tensor) -> Tensor:
		return OccLoss.occupancy_loss(input, target, self.reduction)


	# Loss computed on number of sample points.
	def occupancy_loss(target_sdf, predicted_sdf, reduction='mean'):
		OFFSET = 0.000001

		expanded_target, expanded_predicted = torch.broadcast_tensors(target_sdf, predicted_sdf)

		# Prevent division by 0 by adding a small offset.
		expanded_target = expanded_target + OFFSET
		expanded_predicted = expanded_predicted + OFFSET

		# Get sign of values by dividing by its absolute value.
		target_signs = expanded_target / torch.abs(expanded_target)
		predicted_signs = expanded_predicted / torch.abs(expanded_predicted)

		# Convert signs to binary occupancy values.
		target_occupancy = (target_signs + 1) / 2.0
		predicted_occupancy = (predicted_signs + 1) / 2.0

		# Compute binary cross entropy of occupancy values.
		return nn.functional.binary_cross_entropy(predicted_occupancy, target_occupancy, reduction=reduction)


# Test loss
def test():
	batch_size = 2
	num_points = 2

	target_sdf = torch.rand([batch_size, num_points])
	predicted_sdf = torch.rand([batch_size, num_points])

	reconstruction_loss = ReconstructionLoss()

	print('Reconstruction Loss:')
	print(reconstruction_loss.forward(target_sdf, predicted_sdf))
