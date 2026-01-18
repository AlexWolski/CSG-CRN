import torch
import torch.nn as nn
from losses.proximity_loss import ProximityLoss
from losses.reconstruction_loss import ReconstructionLoss
from utilities.sampler_utils import select_nearest_samples


class Loss(nn.Module):
	TARGET_SAMPLING = "TARGET"
	UNIFIED_SAMPLING = "UNIFIED"
	loss_sampling_methods = [TARGET_SAMPLING, UNIFIED_SAMPLING]


	def __init__(self, loss_metric, num_loss_samples, clamp_dist=None, loss_sampling_method=TARGET_SAMPLING):
		super(Loss, self).__init__()
		self.recon_loss = ReconstructionLoss(loss_metric, clamp_dist)
		self.proximity_loss = ProximityLoss()

		self.num_loss_samples = num_loss_samples
		self.loss_sampling_method = loss_sampling_method


	# Compute reconstruction and primitive loss
	def forward(self, target_near_surface_samples, target_uniform_samples, target_surface_samples, csg_model):
		num_uniform_samples = self.num_loss_samples - target_near_surface_samples.size(1)

		# On refinement iterations, when using Unified sampling, replace the uniform samples with samples near the surface of the current reconstruction.
		if self.loss_sampling_method == self.UNIFIED_SAMPLING:
			target_uniform_samples = self._select_near_surface_samples(target_uniform_samples, num_uniform_samples, csg_model)

		# Compute reconstruction loss
		recon_loss = self.recon_loss(target_near_surface_samples, target_uniform_samples, target_surface_samples, csg_model)

		# Compute primitive loss
		primitive_distances = []
		_ = csg_model.sample_csg(target_surface_samples, out_primitive_samples=primitive_distances)
		proximity_loss = self.proximity_loss(primitive_distances)

		return recon_loss + proximity_loss


	# Select samples from the given SDF point cloud that are near the surface of both the target and reconstruction shapes.
	def _select_near_surface_samples(self, target_uniform_samples, num_uniform_samples, csg_model):
		target_points = target_uniform_samples[..., :3]
		target_distances = target_uniform_samples[..., 3]

		# Compute the reconstruction shape distances and apply a union with the target shape distances.
		combined_distances = csg_model.sample_csg(target_points, initial_distances=target_distances)
		(_, near_surface_points, _, near_surface_indices) = select_nearest_samples(target_points, combined_distances, num_uniform_samples)

		# Select the near-surface points and distances to target shape.
		near_surface_target_distances = torch.gather(target_distances, 1, near_surface_indices)
		near_surface_samples = torch.cat((near_surface_points, near_surface_target_distances.unsqueeze(-1)), dim=-1)

		return near_surface_samples
