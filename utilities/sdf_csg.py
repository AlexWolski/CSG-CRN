import torch
from utilities.sdf_primitives import sdf_ellipsoid, sdf_cuboid, sdf_cylinder


MAX_SDF_VALUE = 1


# Smooth minimum and maximum borrowed from iquilezles.org
# https://iquilezles.org/articles/smin/


def smooth_min(a, b, blending):
	absolute_diff = (a-b).abs()
	h = torch.max(blending - absolute_diff, torch.zeros_like(a)) / blending
	smooth_factor = h * h * blending * 0.25

	return torch.min(a, b) - smooth_factor


def smooth_max(a, b, blending):
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

	def __init__(self, device):
		# List of all primitives and operations to build CSG model
		self.csg_commands = []
		self.device = device


	def add_command(self, shape_weights, operation_weights, translations, rotations, scales, blending, roundness):
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

		# Compute weighted averge distance
		for shape in range(command['shape weights'].size(dim=-1)):
			weight = command['shape weights'][:,shape].unsqueeze(-1)
			distances += weight * CSGModel.sdf_functions[shape](query_points, *command['transforms'], command['roundness'])

		return distances


	def apply_operation(distances, new_distances, command):
		final_distance = 0

		# Compute weighted averge result
		for operation in range(command['operation weights'].size(dim=-1)):
			weight = command['operation weights'][:,operation].unsqueeze(-1)
			final_distance += weight * CSGModel.operation_functions[operation](distances, new_distances, command['blending'])

		return final_distance


	def sample_csg(self, query_points):
		(batch_size, num_points, _) = query_points.size()

		# Set initial SDF to a set maximum value instead of float('inf')
		distances = torch.full((batch_size, num_points), MAX_SDF_VALUE).to(self.device)

		# Compute combined SDF
		for command in self.csg_commands:
			new_distances = CSGModel.sample_sdf(query_points, command)
			distances = CSGModel.apply_operation(distances, new_distances, command)

		return distances


# Test SDFs
def test():
	from torch.distributions.uniform import Uniform

	batch_size = 2
	num_points = 2

	points = Uniform(-0.5, 0.5).sample((batch_size, num_points, 3))

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

	myModel = CSGModel(torch.device('cpu'))
	myModel.add_command(shape_weights1, operation_weights1, translations1, rotations1, scales1, blending1, roundness1)
	myModel.add_command(shape_weights2, operation_weights2, translations2, rotations2, scales2, blending2, roundness2)
	distances = myModel.sample_csg(points)

	print('Weighted SDF Samples:')
	print(distances)