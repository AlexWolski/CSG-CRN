import torch.nn as nn
from losses.proximity_loss import ProximityLoss
from losses.reconstruction_loss import ReconstructionLoss
from losses.spread_loss import SpreadLoss
from utilities.constants import UNIFIED_SAMPLING, TARGET_SAMPLING
from utilities.datasets import NEAR_SURFACE_SAMPLE_FACTOR
from utilities.sampler_utils import select_near_surface_samples


class Loss(nn.Module):
	def __init__(self, loss_metric, num_loss_samples, prims_per_cascade, spread_loss_weight=None, clamp_dist=None, excess_loss_weight=None, loss_sampling_method=TARGET_SAMPLING):
		super(Loss, self).__init__()
		self.recon_loss = ReconstructionLoss(loss_metric, excess_loss_weight, clamp_dist)
		self.proximity_loss = ProximityLoss()
		self.spread_loss = SpreadLoss(prims_per_cascade)
		self.spread_loss_weight = spread_loss_weight

		self.num_loss_samples = num_loss_samples
		self.loss_sampling_method = loss_sampling_method


	# Compute reconstruction and primitive loss
	def forward(self, target_near_surface_samples, target_uniform_samples, target_surface_samples, csg_model):
		# When using Unified sampling, generate near-surface samples by filtering by distance to both the target and reconstruction shapes.
		if self.loss_sampling_method == UNIFIED_SAMPLING:
			num_near_surface_samples = target_near_surface_samples.size(1) // NEAR_SURFACE_SAMPLE_FACTOR
			target_near_surface_samples = select_near_surface_samples(target_near_surface_samples, num_near_surface_samples, csg_model)

		# Compute reconstruction loss
		recon_loss = self.recon_loss(target_near_surface_samples, target_uniform_samples, target_surface_samples, csg_model)

		# Compute primitive loss
		primitive_distances = []
		_ = csg_model.sample_csg(target_surface_samples, out_primitive_samples=primitive_distances)
		proximity_loss = self.proximity_loss(primitive_distances)

		# Compute spread loss
		if self.spread_loss_weight is not None and self.spread_loss_weight > 0:
			spread_loss = self.spread_loss(csg_model)
		else:
			spread_loss = 0

		return recon_loss + proximity_loss + spread_loss

