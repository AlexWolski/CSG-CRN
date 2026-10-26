import torch
import torch.nn as nn


# The demoninator variable in the exponential distance weight term: exp(-|prim_dist| / local_loss_dist).
# As local_loss_dist gets larger, samplers farther away from the measured primitive gain weight until a value of ~2 where all samples have equivalent weight.
# As local_loss_dist gets smaller, samplers closer to the measured primitive gain weight until a value of ~0.01 where only surface samples are measured.
DEFAULT_LOCAL_LOSS_DIST = 0.05
EPSILON = 1e-6


# Weight the reconstruction loss of each sample by proximity of the predicted primitive for each primitive.
class LocalLoss(nn.Module):
	def __init__(self, prims_per_cascade, local_loss_dist=DEFAULT_LOCAL_LOSS_DIST):
		super(LocalLoss, self).__init__()
		self.prims_per_cascade = prims_per_cascade
		self.local_loss_dist = local_loss_dist


	def forward(self, point_loss, primitive_distances, csg_model):
		"""
		Apply a local sample weighting to the reconstruction loss. Computations from the reconstruction loss are required.

		Parameters
		----------
		point_loss : list of torch.Tensor
			Each entry in the list is a primitive, and each tensor contains the per-sample loss for that primitive.
		primitive_samples : list of torch.Tensor
			Each entry in the list is a primitive, and each tensor contains the per-sample signed distance value for that primitive.
		csg_model : CSGModel
			Current reconstruction CSG model.

		"""
		# List of average local reconstruction error for each predicted primitive.
		primitive_local_errors = []

		for command_index in self._trained_command_indices(csg_model):
			# Weigh each loss sample by its distance to the primitive surface: weight 1 on the
			# surface, 1/e at local_los_dist, and near 0 beyond a few multiples of it. The weights
			# form a soft window selecting the samples the primitive is graded on. They are detached
			# so that the only way to reduce the loss is to fix the reconstruction error inside the
			# window, never to shrink or move the window away from poorly fit regions.

			# The weight for each sample is defined as: e^(-|prim_dist|/local_loss_dist)
			# where `prim_dist` is the signed distance from the sample to the primitive
			# and local_loss_dist controls how strictly the weight is applied to samples further away.
			# The gradient is stopped on the weight to prevent workarounds such as predicting smaller primitives.
			with torch.no_grad():
				sample_weights = torch.exp(-primitive_distances[command_index].abs() / self.local_loss_dist)

			# The local error of a primitive is defined as: sum(sample_weight*sample_loss) / sum(sample_weight)
			# We devide by sum(sample_weight) to compute a mean error that normalizes for primitive size.
			local_error = (sample_weights * point_loss).sum(dim=1) / (sample_weights.sum(dim=1) + EPSILON)
			primitive_local_errors.append(local_error)

		if not primitive_local_errors:
			return torch.tensor(0.0, device=point_loss.device)

		# Return average local reconstruction loss for all primitives and all batches.
		return torch.stack(primitive_local_errors).mean()


	# Find the commands currently being trained: those with gradients attached, or the final
	# cascade's commands when gradients are disabled (e.g. during validation).
	def _trained_command_indices(self, csg_model):
		if torch.is_grad_enabled():
			command_indices = [index for index, command in enumerate(csg_model.csg_commands) if command['translations'].requires_grad]

			if command_indices:
				return command_indices

		return list(range(max(csg_model.num_commands - self.prims_per_cascade, 0), csg_model.num_commands))
