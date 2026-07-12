import torch
import torch.nn as nn

from utilities.csg_model import CSGModel

EPSILON = 1e-8

# Penalize newly generated primitives that are too close to existing primitives in the initial reconstruciton.
class SpreadLoss(nn.Module):
	def __init__(self, prims_per_cascade, clamp_dist=0.1):
		super(SpreadLoss, self).__init__()
		# Number of primitives predicted each cascade.
		self.prims_per_cascade = prims_per_cascade
		# Only penalize primitives within `clamp_dist` of an existing primitive in the initial reconstruction.
		self.clamp_dist = clamp_dist

	def forward(self, csg_model):
		num_init_prims = csg_model.num_commands - self.prims_per_cascade

		# Spread loss cannot be computed when there are no primitives in the initial reconstruction.
		if num_init_prims <= 0:
			return 0

		# Extract the position of all primitives in the CSG model.
		translations_list = []

		for command in csg_model.csg_commands:
			translations_list.append(command['translations'])

		translations = torch.stack(translations_list, dim=1)

		# Split the translations for primitives in the initial reconstruction versus newly generated primitives.
		(init_translations, new_translations) = torch.split(translations, [num_init_prims, self.prims_per_cascade], dim=1)

		# translations_list has shape (B, P, 3) where B is batch size and P is num_commands.
		# Broadcast new_translations to shape (B, R, 1, 3) where R is the number of refinement primitives.
		# Broadcase init_translations to shape (B, 1, I 3) where I is the number of initial reconstruciton primitives.
		# The difference will have shape (B, R, I, 3) and contain the distance for all combinations of initial and refinement primitives.
		differences = new_translations.unsqueeze(2) - init_translations.unsqueeze(1)
		# Shape (B, R, I).
		sq_distances = torch.sum(torch.square(differences), dim=-1)

		# Compute the minimum distance between each new primitive and the closest primitive in the initial reconstruciton.
		min_sq_distances = torch.amin(sq_distances, dim=-1)
		min_distances = torch.sqrt(min_sq_distances + EPSILON)

		# Clamp the minimum distances to between 0 and clamp_dist. Invert the values to get loss.
		spread_loss = self.clamp_dist - torch.clamp(min_distances, min=0, max=self.clamp_dist)

		# Return the average spread loss.
		return torch.mean(spread_loss)


# Test loss
def test():
	batch_size = 2

	translations1 = torch.tensor([0,0,0], dtype=torch.float).repeat(batch_size,1)
	rotations1 = torch.tensor([1,0,0,0], dtype=torch.float).repeat(batch_size,1)
	scales1 = torch.tensor([0.6,0.6,0.6], dtype=torch.float).repeat(batch_size,1)
	shape_weights1 = torch.tensor([1,0,0], dtype=torch.float).repeat(batch_size,1)
	operation_weights1 = torch.tensor([1,0], dtype=torch.float).repeat(batch_size,1)
	blending1 = torch.tensor([0.001], dtype=torch.float).repeat(batch_size,1)
	roundness1 = torch.tensor([0], dtype=torch.float).repeat(batch_size,1)

	translations2 = torch.tensor([0.3,0,0], dtype=torch.float).repeat(batch_size,1)
	rotations2 = torch.tensor([1,0,0,0], dtype=torch.float).repeat(batch_size,1)
	scales2 = torch.tensor([0.3,0.3,0.3], dtype=torch.float).repeat(batch_size,1)
	shape_weights2 = torch.tensor([0,1,0], dtype=torch.float).repeat(batch_size,1)
	operation_weights2 = torch.tensor([0,1], dtype=torch.float).repeat(batch_size,1)
	blending2 = torch.tensor([0.001], dtype=torch.float).repeat(batch_size,1)
	roundness2 = torch.tensor([0], dtype=torch.float).repeat(batch_size,1)

	myModel = CSGModel(batch_size)
	myModel.add_command(shape_weights1, operation_weights1, translations1, rotations1, scales1, blending1, roundness1)
	myModel.add_command(shape_weights2, operation_weights2, translations2, rotations2, scales2, blending2, roundness2)

	spread_loss = SpreadLoss(prims_per_cascade=1)

	print('Spread Loss:')
	print(spread_loss.forward(myModel))
