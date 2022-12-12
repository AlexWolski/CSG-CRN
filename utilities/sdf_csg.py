import math
import torch
from torch.distributions.uniform import Uniform
from utilities.sdf_primitives import sdf_ellipsoid, sdf_cuboid, sdf_cylinder


# Supported bounds is a sphere with radius MAX_BOUND
MAX_BOUND = 0.5
# Maximum SDF value is twice the radius
MAX_SDF_VALUE = MAX_BOUND * 2
# Multiplier to estimate number of uniform points needed to sample near-surface points
SURFACE_SAMPLE_RATIO = 5


# Smooth minimum and maximum borrowed from iquilezles.org
# https://iquilezles.org/articles/smin/


def smooth_min(a, b, blending):
	if blending is None:
		smooth_factor = 0
	else:
		absolute_diff = (a-b).abs()
		h = torch.max(blending - absolute_diff, torch.zeros_like(a)) / blending
		smooth_factor = h * h * blending * 0.25

	return torch.min(a, b) - smooth_factor


def smooth_max(a, b, blending):
	if blending is None:
		smooth_factor = 0
	else:
		absolute_diff = (a-b).abs()
		h = torch.max(blending - absolute_diff, torch.zeros_like(a)) / blending
		smooth_factor = h * h * blending * 0.25

	return torch.max(a, b) + smooth_factor


# Union of two SDFs
def add_sdf(distances, new_distances, blending):
	return smooth_min(distances, new_distances, blending)


# Intersection of one SDF and the conjugate of the other
def subtract_sdf(distances, new_distances, blending):
	return smooth_max(distances, -new_distances, blending)


class CSGModel():
	sdf_functions = [
		sdf_ellipsoid,
		sdf_cuboid,
		sdf_cylinder
	]

	operation_functions = [
		add_sdf,
		subtract_sdf
	]

	def __init__(self, device=torch.device('cpu')):
		# List of all primitives and operations to build CSG model
		self.csg_commands = []
		self.device = device


	# Add a primitive to the CSG model
	def add_command(self, shape_weights, operation_weights, translations, rotations, scales, blending=None, roundness=None):
		self.csg_commands.append({
			'shape weights': shape_weights,
			'operation weights': operation_weights,
			'transforms': (translations, rotations, scales),
			'blending': blending,
			'roundness': roundness
		})


	# Compute blended SDF for all primitive types given primitive weights and a transform
	# We blend the weighted primitives instead of 
	def sample_sdf(query_points, command):
		distances = 0

		if command['roundness'] is None:
			roundness = 0
		else:
			roundness = command['roundness']

		# Compute weighted averge distance
		for shape in range(command['shape weights'].size(dim=-1)):
			weight = command['shape weights'][:,shape].unsqueeze(-1)
			distances += weight * CSGModel.sdf_functions[shape](query_points, *command['transforms'], roundness)

		return distances


	# Combine a primitive with the CSG model using a boolean operation
	def apply_operation(distances, new_distances, command):
		final_distance = 0

		# Compute weighted averge result
		for operation in range(command['operation weights'].size(dim=-1)):
			weight = command['operation weights'][:,operation].unsqueeze(-1)
			final_distance += weight * CSGModel.operation_functions[operation](distances, new_distances, command['blending'])

		return final_distance


	# Sample signed distances from a set of query points
	def sample_csg(self, query_points):
		(batch_size, num_points, _) = query_points.size()

		# Set initial SDF to a set maximum value instead of float('inf')
		distances = torch.full((batch_size, num_points), MAX_SDF_VALUE).to(self.device)

		# Compute combined SDF
		for command in self.csg_commands:
			new_distances = CSGModel.sample_sdf(query_points, command)
			distances = CSGModel.apply_operation(distances, new_distances, command)

		return distances


	# Sample a given number of signed distances at uniformly distributed points
	def sample_csg_uniform(self, batch_size, num_points):
		uniform_points = Uniform(-MAX_BOUND, MAX_BOUND).sample((batch_size, num_points, 3)).to(self.device)
		uniform_distances = self.sample_csg(uniform_points)

		return (uniform_points, uniform_distances)


	# Sample a given number of signed distances at near-surface points
	def sample_csg_surface(self, batch_size, num_points, sample_dist):
		# Get uniform points
		num_uniform_points = math.ceil(num_points / sample_dist) * SURFACE_SAMPLE_RATIO
		(uniform_points, uniform_distances) = self.sample_csg_uniform(batch_size, num_uniform_points)

		# Store all indices in flat tensor
		all_indices = None

		for batch in range(batch_size):
			# Select indices for near-surface points
			mask = (abs(uniform_distances[batch]) <= sample_dist)
			indices = mask.nonzero()

			# If there are too few near-surface points, mix in uniform points
			if len(indices) < num_points:
				num_uniform_indices = num_points - len(indices)
				unifrom_indices = torch.randint(num_uniform_points, (num_uniform_indices, 1)).to(self.device)
				indices = torch.cat((indices, unifrom_indices))
			# Otherwise slice the needed number of points
			else:
				indices = indices[:num_points]

			# Adjust index positions based on batch
			indices = torch.add(indices, batch * num_points)

			# Add indices to total
			if all_indices is None:
				all_indices = indices
			else:
				all_indices = torch.cat((all_indices, indices))

		# Flatten sample tensors
		uniform_points = uniform_points.view(-1, 3)
		uniform_distances = uniform_distances.view(-1, 1)
		# Index flattened tensors
		uniform_points = uniform_points[all_indices]
		uniform_distances = uniform_distances[all_indices]
		# Reshape
		uniform_points = uniform_points.view(batch_size, num_points, 3)
		uniform_distances = uniform_distances.view(batch_size, num_points)

		return (uniform_points, uniform_distances)


# Test SDFs
def test():
	batch_size = 2
	num_points = 1000

	translations1 = torch.tensor([0,0,0], dtype=float).repeat(batch_size,1)
	rotations1 = torch.tensor([1,0,0,0], dtype=float).repeat(batch_size,1)
	scales1 = torch.tensor([0.6,0.6,0.6], dtype=float).repeat(batch_size,1)
	shape_weights1 = torch.tensor([1,0,0], dtype=float).repeat(batch_size,1)
	operation_weights1 = torch.tensor([1,0], dtype=float).repeat(batch_size,1)
	blending1 = torch.tensor([0.001], dtype=float).repeat(batch_size,1)
	roundness1 = torch.tensor([0], dtype=float).repeat(batch_size,1)

	translations2 = torch.tensor([0.3,0,0], dtype=float).repeat(batch_size,1)
	rotations2 = torch.tensor([1,0,0,0], dtype=float).repeat(batch_size,1)
	scales2 = torch.tensor([0.3,0.3,0.3], dtype=float).repeat(batch_size,1)
	shape_weights2 = torch.tensor([0,1,0], dtype=float).repeat(batch_size,1)
	operation_weights2 = torch.tensor([0,1], dtype=float).repeat(batch_size,1)
	blending2 = torch.tensor([0.001], dtype=float).repeat(batch_size,1)
	roundness2 = torch.tensor([0], dtype=float).repeat(batch_size,1)

	myModel = CSGModel()
	myModel.add_command(shape_weights1, operation_weights1, translations1, rotations1, scales1, blending1, roundness1)
	myModel.add_command(shape_weights2, operation_weights2, translations2, rotations2, scales2, blending2, roundness2)
	(points, distances) = myModel.sample_csg_surface(batch_size, num_points, 0.1)

	print('Sample points:')
	print(points)
	print('Weighted distances:')
	print(distances)