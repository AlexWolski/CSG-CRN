import torch
import torch.nn as nn
from losses.proximity_loss import ProximityLoss
from losses.reconstruction_loss import ReconstructionLoss
from losses.spread_loss import SpreadLoss
from utilities.constants import UNIFIED_SAMPLING, TARGET_SAMPLING
from utilities.csg_model import CSGModel, subtract_sdf
from utilities.datasets import NEAR_SURFACE_SAMPLE_FACTOR
from utilities.sampler_utils import select_near_surface_samples


class Loss(nn.Module):
	def __init__(self, loss_metric, num_loss_samples, prims_per_cascade, spread_loss_weight=None, clamp_dist=None, excess_loss_weight=None, loss_sampling_method=TARGET_SAMPLING, residual_only_training=False):
		super(Loss, self).__init__()
		self.recon_loss = ReconstructionLoss(loss_metric, excess_loss_weight, clamp_dist)
		self.proximity_loss = ProximityLoss()
		self.spread_loss = SpreadLoss(prims_per_cascade)
		self.spread_loss_weight = spread_loss_weight
		self.prims_per_cascade = prims_per_cascade
		self.residual_only_training = residual_only_training

		self.num_loss_samples = num_loss_samples
		self.loss_sampling_method = loss_sampling_method


	# Compute reconstruction and primitive loss
	def forward(self, target_near_surface_samples, target_uniform_samples, target_surface_samples, csg_model):
		recon_csg_model = csg_model

		# When training on the residual, only compute the loss between the newly generated primitives and the missing portion of the target volume.
		if self.residual_only_training:
			num_init_prims = csg_model.num_commands - self.prims_per_cascade

			if num_init_prims > 0:
				init_commands = csg_model.csg_commands[:num_init_prims]

				# Compute the residual volume by subtracting the initial reconstruction from the target volume.
				target_sample_split = (target_near_surface_samples.size(1), target_uniform_samples.size(1))
				target_samples = torch.cat((target_near_surface_samples, target_uniform_samples), dim=1)

				init_recon_sdf = CSGModel.sample_csg_commands(target_samples[..., :3], init_commands)
				residual_sdf = subtract_sdf(target_samples[..., 3], init_recon_sdf, blending=0.2)
				residual_samples = torch.cat((target_samples[..., :3], residual_sdf.unsqueeze(-1)), dim=-1)

				(target_near_surface_samples, target_uniform_samples) = residual_samples.split(target_sample_split, dim=1)

				# Store the newly generated primitives in a separate CSG model.
				recon_csg_model = CSGModel(csg_model.batch_size, csg_model.device)
				recon_csg_model.csg_commands = csg_model.csg_commands[num_init_prims:]
				recon_csg_model.num_commands = self.prims_per_cascade

		# When using Unified sampling, generate near-surface samples by filtering by distance to both the target and reconstruction shapes.
		if self.loss_sampling_method == UNIFIED_SAMPLING:
			num_near_surface_samples = target_near_surface_samples.size(1) // NEAR_SURFACE_SAMPLE_FACTOR
			target_near_surface_samples = select_near_surface_samples(target_near_surface_samples, num_near_surface_samples, recon_csg_model)

		# Compute reconstruction loss
		recon_loss = self.recon_loss(target_near_surface_samples, target_uniform_samples, target_surface_samples, recon_csg_model)

		# Compute primitive loss
		primitive_distances = []
		_ = csg_model.sample_csg(target_surface_samples, out_primitive_samples=primitive_distances)
		proximity_loss = self.proximity_loss(primitive_distances)

		# Compute spread loss
		if self.spread_loss_weight is not None and self.spread_loss_weight > 0:
			spread_loss = self.spread_loss_weight * self.spread_loss(csg_model)
		else:
			spread_loss = 0

		return recon_loss + proximity_loss + spread_loss

