import math
import torch
from torch.distributions.uniform import Uniform
from utilities.sdf_primitives import sdf_ellipsoid, sdf_cuboid, sdf_cylinder, world_to_local_points


# Supported bounds is a cube with length MAX_BOUND - MIN_BOUND
MIN_BOUND = -1
MAX_BOUND = 1
# Maximum SDF value is twice the radius
MAX_SDF_VALUE = MAX_BOUND * 2


# Smooth minimum and maximum borrowed from iquilezles.org
# https://iquilezles.org/articles/smin/


def smooth_min(a, b, blending):
	if blending is None:
		smooth_factor = 0
	else:
		absolute_diff = (a-b).abs()
		h = torch.clamp(blending - absolute_diff, min=0) / blending
		smooth_factor = h * h * blending * 0.25

	return torch.min(a, b) - smooth_factor


def smooth_max(a, b, blending):
	return -smooth_min(-a, -b, blending)


# Union of two SDFs
def add_sdf(distances, new_distances, blending):
	return smooth_min(distances, new_distances, blending)


# Intersection of one SDF and the conjugate of the other
def subtract_sdf(distances, new_distances, blending):
	return smooth_max(distances, -new_distances, blending)


# Map primitive SDF function to MagicaCSG primitive name
primitive_name_map = {
	sdf_ellipsoid: "sphere",
	sdf_cuboid: "cube",
	sdf_cylinder: "cylinder"
}


# Map SDF operation function to MagicaCSG operation name
operation_name_map = {
	add_sdf: "add",
	subtract_sdf: "sub",
}


# Return the MagicaCSG primitive name given a shape weight tensor
def get_primitive_name(shape_weights):
	shape_index = torch.argmax(shape_weights).item()
	shape_function = CSGModel.sdf_functions[shape_index]
	return primitive_name_map[shape_function]


# Return the MagicaCSG operation name given a shape weight tensor
def get_operation_name(operation_weights):
	operation_index = torch.argmax(operation_weights).item()
	operation_function = CSGModel.operation_functions[operation_index]
	return operation_name_map[operation_function]


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

	num_shapes = len(sdf_functions)
	num_operations = len(operation_functions)


	def __init__(self, batch_size, device=torch.device('cpu')):
		# List of all primitives and operations to build CSG model
		self.csg_commands = []
		self.num_commands = 0
		self.batch_size = batch_size
		self.device = device


	def clone(self):
		cloned_model = CSGModel(self.batch_size, self.device)

		for command in self.csg_commands:
			cloned_model.add_command(
				command['shape weights'].clone(),
				command['operation weights'].clone(),
				command['translations'].clone(),
				command['rotations'].clone(),
				command['scales'].clone(),
				command['blending'],
				command['roundness']
			)

		return cloned_model


	def detach(self):
		for command_index in range(self.num_commands):
			command_list = self.csg_commands[command_index]

			for command_key, command in command_list.items():
				self.csg_commands[command_index][command_key] = command.detach() if torch.is_tensor(command) else command

		return self


	# Validate that the given command has the correct batch size
	def _validate_batch_size(self, command):
		batch_sizes_set = set()

		for _key, value in command.items():
			if value != None:
				batch_sizes_set.add(value.size(0))

		# Ensure that all the tensors in the command have the same batch size
		if len(batch_sizes_set) > 1:
			raise Exception(f'All tensors in a command must must have the same batch size. Found batch sizes: {batch_sizes_set}')

		batch_size = batch_sizes_set.pop()

		# Set the number of batches if this is the first command.
		# Ensure that the batch size of the command matches the existing commands.
		if self.batch_size != batch_size:
			raise Exception(f'Expected batch size of {self.batch_size} but was given a command with batch size of {batch_size}.')


	# Add a primitive to the CSG model
	def add_command(self, shape_weights, operation_weights, translations, rotations, scales, blending=None, roundness=None):
		command = {
			'shape weights': shape_weights,
			'operation weights': operation_weights,
			'translations': translations,
			'rotations': rotations,
			'scales': scales,
			'blending': blending,
			'roundness': roundness
		}

		self._validate_batch_size(command)
		self.csg_commands.append(command)
		self.num_commands += 1


	# Add batches of commands from another other CSG model to this model
	def add_batches_from_csg_model(self, other_csg_model):
		if self.device != other_csg_model.device:
			raise Exception(f'CSG Model devices do not match. Expected {self.device} but found {other_csg_model.device}')

		if self.csg_commands and self.num_commands != other_csg_model.num_commands:
			raise Exception(f'CSG Model command lengths do not match. Expected {self.num_commands} but found {other_csg_model.num_commands}')

		# Copy commands if this model doesn't have any
		if not self.csg_commands:
			self.csg_commands = other_csg_model.csg_commands
			self.batch_size = other_csg_model.batch_size
			self.num_commands = other_csg_model.num_commands
			return

		# Combine CSG commands
		for command_index in range(self.num_commands):
			other_command_list = other_csg_model.csg_commands[command_index]

			for command_key, other_command in other_command_list.items():
				command = self.csg_commands[command_index][command_key]

				# Handle case where one or both of the commands are None
				if command == None and other_command == None:
					self.csg_commands[command_index][command_key] = None
				elif command == None or other_command == None:
					raise Exception(f'Cannot combine command {command_key} with \'None\'')
				# Concatenate commands if neither are None
				else:
					self.csg_commands[command_index][command_key] = torch.cat((command, other_command))

		self.batch_size += other_csg_model.batch_size


	# Compute blended SDF for all primitive types given primitive weights and a transform
	def sample_sdf(query_points, command):
		distances = 0

		if command['roundness'] is None:
			roundness = 0
		else:
			roundness = command['roundness']

		# Transform query points
		transformed_query_points = world_to_local_points(query_points, command['translations'], command['rotations'])

		# Compute weighted average distance
		for shape in range(command['shape weights'].size(dim=-1)):
			weight = command['shape weights'][:,shape].unsqueeze(-1)
			distances += weight * CSGModel.sdf_functions[shape](transformed_query_points, command['scales'], roundness)

		return distances


	# Combine a primitive with the CSG model using a boolean operation
	def apply_operation(distances, new_distances, command):
		final_distance = 0

		# Compute weighted average result
		for operation in range(command['operation weights'].size(dim=-1)):
			weight = command['operation weights'][:,operation].unsqueeze(-1)
			final_distance += weight * CSGModel.operation_functions[operation](distances, new_distances, command['blending'])

		return final_distance


	# Sample signed distances from a set of query points given a list of CSG commands
	# Optional out_primitive_samples parameter is a list that gets populated with primitive SDF distances
	def sample_csg_commands(query_points, csg_commands, initial_distances=None, out_primitive_samples=None):
		# Return None if there are no csg commands
		if not csg_commands:
			return None

		(batch_size, num_points, _) = query_points.size()

		# Set initial SDF to a set maximum value instead of float('inf')
		if initial_distances is not None:
			distances = initial_distances
		else:
			distances = torch.full((batch_size, num_points), MAX_SDF_VALUE, device=query_points.device)

		# Compute combined SDF
		for command in csg_commands:
			new_distances = CSGModel.sample_sdf(query_points, command)

			# Populate out parameter
			if out_primitive_samples is not None:
				out_primitive_samples.append(new_distances)

			distances = CSGModel.apply_operation(distances, new_distances, command)

		return distances


	# Sample signed distances from a set of query points
	# Optional out_primitive_samples parameter is a list that gets populated with primitive SDF distances
	def sample_csg(self, query_points, initial_distances=None, out_primitive_samples=None):
		return CSGModel.sample_csg_commands(query_points, self.csg_commands, initial_distances=initial_distances, out_primitive_samples=out_primitive_samples)


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

	points = Uniform(MIN_BOUND, MAX_BOUND).sample((batch_size, num_points, 3)).to(myModel.device)
	distances = myModel.sample_csg(points)

	print('Sample points:')
	print(points)
	print('Weighted distances:')
	print(distances)