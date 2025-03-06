import torch
import torch.nn as nn


# Compute the reconstruction loss between SDFs sampled from two point clouds
# Loss function borrowed from DeepSDF
# https://github.com/facebookresearch/DeepSDF/blob/main/train_deep_sdf.py#L501
class ReconstructionLoss(nn.Module):
	L1_LOSS_FUNC = "L1"
	MSE_LOSS_FUNC = "MSE"
	LOG_LOSS_FUNC = "LOG"

	def __init__(self, loss_metric):
		super(ReconstructionLoss, self).__init__()

		# Select loss function
		match loss_metric:
			case self.L1_LOSS_FUNC:
				self.loss_func = torch.nn.L1Loss(reduction='mean')
			case self.MSE_LOSS_FUNC:
				self.loss_func = torch.nn.MSELoss(reduction='mean')
			case self.LOG_LOSS_FUNC:
				self.loss_func = log_loss
			case None:
				raise Exception("A loss function must be provided")
			case _:
				raise Exception(f"Invalid loss metric: {loss_metric}")


	# Compute average L1 loss of SDF samples of all batches
	def forward(self, target_sdf, predicted_sdf):
		return self.loss_func(target_sdf, predicted_sdf)


	# Log loss function roughly intersecting (0,0)
	def log_loss(target_sdf, predicted_sdf):
		X_OFFSET = 0.01
		Y_OFFSET = 1
		MULTIPLE = 5

		expanded_target, expanded_predicted = torch.broadcast_tensors(target_sdf, predicted_sdf)
		absolute_error = torch.abs(expanded_target - predicted_sdf)
		log_loss = torch.log(absolute_error + X_OFFSET) / MULTIPLE + Y_OFFSET
		return torch.mean(log_loss)


# Test loss
def test():
	batch_size = 2
	num_points = 2

	target_sdf = torch.rand([batch_size, num_points])
	predicted_sdf = torch.rand([batch_size, num_points])

	reconstruction_loss = ReconstructionLoss()

	print('Reconstruction Loss:')
	print(reconstruction_loss.forward(target_sdf, predicted_sdf))
