import torch
import sdf_primitives


MAX_SDF_VALUE = 1


def add_sdf(distances, new_distances):
	return torch.min(distances, new_distances)


def subtract_sdf(distances, new_distances):
	return torch.max(distances, -new_distances)


class CSGModel():
	sdf_functions = [
		sdf_primitives.sdf_ellipsoid,
		sdf_primitives.sdf_cuboid,
		sdf_primitives.sdf_cylinder
	]

	operation_functions = [
		add_sdf,
		subtract_sdf
	]

	def __init__(self, batch_size):

		# Primitives are added and sampled in batches
		self.batch_size = batch_size
		# List of all primitives and operations to build CSG model
		self.csg_commands = []


	def add_sdf(self, shape_weights, operation_weights, translations, rotations, scales, blending):
		self.csg_commands.append({
			'shape weights': shape_weights,
			'operation weights': operation_weights,
			'transforms': (translations, rotations, scales),
			'blending': blending
		})


	# Compute blended SDF for all primitive types given primitive weights and a transform
	# We blend the weighted primitives instead of 
	def sample_sdf(query_points, shape_weights, transforms):
		distances = 0

		# Compute weighted averge distance
		for shape in range(len(shape_weights)):
			distances += shape_weights[shape] * CSGModel.sdf_functions[shape](query_points, *transforms)

		return distances


	def apply_operation(distances, new_distances, operation_weights, blending):
		final_distance = 0

		# Compute weighted averge result
		for operation in range(len(operation_weights)):
			final_distance += operation_weights[operation] * CSGModel.operation_functions[operation](distances, new_distances)

		return final_distance


	def sample_csg(self, query_points):
		# Set initial SDF to a set maximum value instead of float('inf')
		distances = torch.full((self.batch_size, 1), MAX_SDF_VALUE)

		# Compute combined SDF
		for command in self.csg_commands:
			new_distances = CSGModel.sample_sdf(query_points, command['shape weights'], command['transforms'])
			distances = CSGModel.apply_operation(distances, new_distances, command['operation weights'], command['blending'])

		return distances


# Test SDFs
if __name__ == "__main__":
	batch_size = 2
	num_points = 2

	points = torch.rand([batch_size, num_points, 3])

	translations1 = torch.tensor([0,0,0], dtype=float).unsqueeze(0)
	rotations1 = torch.tensor([1,0,0,0], dtype=float).unsqueeze(0)
	scales1 = torch.tensor([0.8,0.8,0.8], dtype=float).unsqueeze(0)
	shape_weights1 = [1,0,0]
	operation_weights1 = [1,0]
	blending1 = 0

	translations2 = torch.tensor([0.8,0,0], dtype=float).unsqueeze(0)
	rotations2 = torch.tensor([1,0,0,0], dtype=float).unsqueeze(0)
	scales2 = torch.tensor([0.3,0.3,0.3], dtype=float).unsqueeze(0)
	shape_weights2 = [0,1,0]
	operation_weights2 = [0,1]
	blending2 = 0

	myModel = CSGModel(batch_size)
	myModel.add_sdf(shape_weights1, operation_weights1, translations1, rotations1, scales1, blending1)
	myModel.add_sdf(shape_weights2, operation_weights2, translations2, rotations2, scales2, blending2)
	distances = myModel.sample_csg(points)

	print('Weighted SDF Samples:')
	print(distances)