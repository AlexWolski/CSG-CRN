import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from utilities.csg_model import CSGModel
from utilities.sampler_utils import sample_points_csg_surface_differentiable
from utilities.accuracy_metrics import compute_chamfer_distance


# Compute the reconstruction loss between SDFs sampled from two point clouds
# Loss function borrowed from DeepSDF
# https://github.com/facebookresearch/DeepSDF/blob/main/train_deep_sdf.py#L501
class ReconstructionLoss(nn.Module):
	L1_LOSS_FUNC = "L1"
	MSE_LOSS_FUNC = "MSE"
	LOG_LOSS_FUNC = "LOG"
	SIG_LOSS_FUNC = "SIG"
	OCC_LOSS_FUNC = "OCC"
	CHAMFER_LOSS_FUNC = "CHAMFER"
	loss_metrics = [L1_LOSS_FUNC, MSE_LOSS_FUNC, LOG_LOSS_FUNC, SIG_LOSS_FUNC, OCC_LOSS_FUNC, CHAMFER_LOSS_FUNC]


	def __init__(self, loss_metric, excess_loss_weight=None, clamp_dist=None):
		super(ReconstructionLoss, self).__init__()

		self.clamp_dist = clamp_dist
		self.loss_metric = loss_metric
		self.excess_loss_weight = excess_loss_weight

		# Select loss function
		match loss_metric:
			case self.L1_LOSS_FUNC:
				self.loss_func = torch.nn.L1Loss(reduction='none')
			case self.MSE_LOSS_FUNC:
				self.loss_func = torch.nn.MSELoss(reduction='none')
			case self.LOG_LOSS_FUNC:
				self.loss_func = LogLoss(reduction='none')
			case self.SIG_LOSS_FUNC:
				self.loss_func = SigLoss(reduction='none')
			case self.OCC_LOSS_FUNC:
				self.loss_func = OccLoss(reduction='none')
			case self.CHAMFER_LOSS_FUNC:
				self.loss_func = ChamferLoss(reduction='none')
			case None:
				raise Exception("A loss function must be provided")
			case _:
				raise Exception(f"Invalid loss metric: {loss_metric}")


	def forward(self, target_near_surface_samples, target_uniform_samples, target_surface_samples, csg_model, out_point_loss=None, out_primitive_samples=None):
		"""
		Compute average loss of SDF samples of all batches

		Parameters
		----------
		target_near_surface_samples : torch.Tensor
			Target shape SDF point cloud tensor for configured loss metric.
		target_uniform_samples : torch.Tensor
			Reconstruction shape SDF point cloud tensor for configured loss metric.
		target_surface_samples : torch.Tensor
			Target shape surface samples.
		csg_model : CSGModel
			Current reconstruction CSG model.
		out_point_loss : list of torch.Tensor
			Optional list output for the per-sample loss values of the reconstruction, computed before clamping and excess weighting.
		out_primitive_samples : list of torch.Tensor
			Optional list output for the SDF distances of each primitive.

		Returns
		-------
		torch.Tensor
			When batch_size is None, returns Float Tensor of size (N, 3) where N=`num_points`.
			When batch_size (B) is provided, returns Float Tensor of size (B, N, 3) where N=`num_points`.
			Points uniformly sampled from a cube with side length `side_length`.

		"""
		# The Chamfer distance metric does not require uniform samples, but does require near-surface samples and a CSG model.
		if self.loss_metric == self.CHAMFER_LOSS_FUNC:
			recon_loss = self.loss_func(target_surface_samples, csg_model)
			return torch.mean(recon_loss)
		# All other loss metrics require a mix of near-surface and uniform samples.
		else:
			(unclamped_target_sdf, unclamped_predicted_sdf) = self.cat_samples(target_near_surface_samples, target_uniform_samples, csg_model, out_primitive_samples)

			# Clamp the SDF values to have a maximum difference of clamp_dist*2.
			if self.clamp_dist != None:
				target_sdf = torch.clamp(unclamped_target_sdf, min=-self.clamp_dist, max=self.clamp_dist)
				predicted_sdf = torch.clamp(unclamped_predicted_sdf, min=-self.clamp_dist, max=self.clamp_dist)
			else:
				target_sdf = unclamped_target_sdf
				predicted_sdf = unclamped_predicted_sdf

			# Compute reconstruction loss with clamp settings.
			point_loss = self.loss_func(target_sdf, predicted_sdf)

			# Return the unclamped loss. When clamping is disabled, the computed loss is already unclamped.
			if out_point_loss is not None:
				unclamped_loss = point_loss if self.clamp_dist == None else self.loss_func(unclamped_target_sdf, unclamped_predicted_sdf)
				out_point_loss.append(unclamped_loss)

			if self.excess_loss_weight == None or self.excess_loss_weight == 1.0:
				# Unweighted loss
				return torch.mean(point_loss)
			else:
				# Increase the loss for samples detecting excess volume.
				# (The sample point is outside both the target and reconstruction volumes, but the reconstruction is closer.)
				excess_samples_mask = (target_sdf > 0.0) & (predicted_sdf > 0.0) & (target_sdf - predicted_sdf > 0)
				sample_weights = torch.where(excess_samples_mask, self.excess_loss_weight, 1.0)
				recon_loss = torch.mean(point_loss * sample_weights)
				return recon_loss


	def cat_samples(self, target_near_surface_samples, target_uniform_samples, csg_model, out_primitive_samples=None):
		target_sdf_samples = torch.cat((target_near_surface_samples, target_uniform_samples), 1)
		target_points = target_sdf_samples[..., :3]
		target_sdf = target_sdf_samples[..., 3]

		predicted_sdf = csg_model.sample_csg(target_points, out_primitive_samples=out_primitive_samples)

		return (target_sdf, predicted_sdf)


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


# Compute loss on the number of samples that are within a narrow margin of the target sample
class SigLoss(_Loss):
	def __init__(self, reduction: str = 'mean'):
		super(SigLoss, self).__init__(None, None, reduction)


	def forward(self, input: Tensor, target: Tensor) -> Tensor:
		return SigLoss.sigmoid_loss(input, target, self.reduction)
	

	def sigmoid_loss(target_sdf, predicted_sdf, reduction='mean'):
		SIGMOID_EXPONENT = -500
		X_OFFSET = 5

		expanded_target, expanded_predicted = torch.broadcast_tensors(target_sdf, predicted_sdf)
		absolute_error = torch.abs(expanded_target - predicted_sdf)
		sig_loss = 1 / (1 + torch.exp(SIGMOID_EXPONENT * absolute_error + X_OFFSET))
		sig_l1_loss = (sig_loss + absolute_error) / 2

		match reduction:
			case 'none':
				return sig_l1_loss
			case 'sum':
				return torch.sum(sig_l1_loss)
			case 'mean':
				return torch.mean(sig_l1_loss)


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



# Chamfer distance loss function
class ChamferLoss(_Loss):
	RESOLUTION = 64


	def __init__(self, reduction: str = 'mean'):
		super(ChamferLoss, self).__init__(None, None, reduction)


	def forward(self, input: Tensor, target: CSGModel) -> Tensor:
		return ChamferLoss.chamfer_loss(input, target, self.RESOLUTION, self.reduction)


	def chamfer_loss(target_surface_samples, csg_model, resolution, reduction='mean'):
		num_samples = target_surface_samples.size(1)
		surface_points = sample_points_csg_surface_differentiable(csg_model, resolution, num_samples)
		chamfer_dist = compute_chamfer_distance(target_surface_samples[:,:surface_points.size(1)], surface_points, no_grad=False)

		match reduction:
			case 'none':
				return chamfer_dist
			case 'sum':
				return torch.sum(chamfer_dist)
			case 'mean':
				return torch.mean(chamfer_dist)


# Test loss
def test():
	batch_size = 2
	num_points = 2

	target_sdf = torch.rand([batch_size, num_points])
	predicted_sdf = torch.rand([batch_size, num_points])

	reconstruction_loss = ReconstructionLoss()

	print('Reconstruction Loss:')
	print(reconstruction_loss.forward(target_sdf, predicted_sdf))
