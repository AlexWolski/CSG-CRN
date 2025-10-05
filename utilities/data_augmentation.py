import argparse
import numpy as np
import random
import torch
import math
from enum import Enum
from utilities.point_transform import rotate_point_cloud, rotate_point_cloud_batch
from scipy.spatial.transform import Rotation


class RotationAxis(Enum):
	xAxis = 'X'
	yAxis = 'Y'
	zAxis = 'Z'
	random = 'RANDOM'
	allAxes = 'ALL'

	# Make constructor case-insensitive
	@classmethod
	def _missing_(self, value):
		validValues = [str(x) for x in list(RotationAxis)]
		valueUpper = value.upper()

		if valueUpper not in validValues:
			raise ValueError()

		return self(valueUpper)

	def __str__(self):
		return self.value

	# Randomly select one of the 3 concrete enum states X, Y, and Z
	@classmethod
	def random_value(self):
		index = random.randrange(3)
		return list(RotationAxis)[index]


class ScaleAxis(Enum):
	xAxis = 'X'
	yAxis = 'Y'
	zAxis = 'Z'
	random = 'RANDOM'
	allAxes = 'ALL'

	# Make constructor case-insensitive
	@classmethod
	def _missing_(self, value):
		validValues = [str(x) for x in list(ScaleAxis)]
		valueUpper = value.upper()

		if valueUpper not in validValues:
			raise Exception(f"Cannot construct ScaleAxis enum with invalid value: '{valueUpper}'\nValid values are: {validValues}")

		return self(valueUpper)

	def __str__(self):
		return self.value

	# Randomly select one of the 3 concrete enum states X, Y, and Z
	@classmethod
	def random_value(self):
		index = random.randrange(3)
		return list(ScaleAxis)[index]


def get_augment_parser(group_name='AUGMENT SETTINGS', suppress_default=False):
	argument_default = argparse.SUPPRESS if suppress_default else None
	parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS, argument_default=argument_default)
	parser_group = parser.add_argument_group(group_name)

	parser_group.add_argument('--augment_data', default=False, action='store_true', help='Enable augmentation of object samples with random rotation, scaling, and noise')
	parser_group.add_argument('--augment_copies', type=int, default=1, help='Number of augmented copies of each object to create')
	parser_group.add_argument('--no_rotation', default=False, action='store_true', help='Disable rotations in data augmentation')
	parser_group.add_argument('--no_noise', default=False, action='store_true', help='Disable gaussian noise for sample points and distances')
	parser_group.add_argument('--rotate_axis', default='ALL', type=RotationAxis, choices=list(RotationAxis), help='Axis to rotate around')
	parser_group.add_argument('--noise_variance', type=float, default=1e-10, help='The variance of the gaussian noise aded to each sample point and distance')

	return parser


# Convert a SciPy Rotation object to a quaternion PyTorch tensor of format [w, x, y, z]
def rotation_to_quat_tensor(rotation):
	return torch.from_numpy(rotation.as_quat(scalar_first=True).astype(np.float32))


# Generate a random rotation quaternion
def random_rotation(rotate_axis):
	# Generate a quaternion with random rotations on all axes
	if rotate_axis == RotationAxis('ALL'):
		random_rot = rotation_to_quat_tensor(Rotation.random())
		return random_rot

	# Select an axis to rotate around
	if rotate_axis == RotationAxis('RANDOM'):
		rotate_axis = RotationAxis.random_value()

	# Generate a random rotation
	random_rot = Rotation.from_euler(rotate_axis.value, random.uniform(0, 360), degrees=True)
	random_rot = rotation_to_quat_tensor(random_rot)

	return random_rot


# Generate a random rotation quaternion
def random_rotation_batch(rotate_axis, batch_size):
	rotation_list = [random_rotation(rotate_axis) for i in range(batch_size)]
	return torch.stack(rotation_list, dim=0)


# Generate a random scale vector
def random_scale(scale_axis, min_scale, max_scale):
	# Generate a random value for each axis
	if scale_axis == ScaleAxis('ALL'):
		scale_vec = [random.uniform(min_scale, max_scale) for i in range(3)]
		return torch.FloatTensor(scale_vec)

	# Select an axis to scale
	if scale_axis == ScaleAxis('RANDOM'):
		scale_axis = ScaleAxis.random_value()

	# Generate a random scalar for each axis
	axes = [ScaleAxis('X'), ScaleAxis('Y'), ScaleAxis('Z')]
	scale_vec = [random.uniform(min_scale, max_scale)
				if scale_axis == axes[i] else 1.0 for i in range(3)]

	return torch.FloatTensor(scale_vec)


# Generate a random scale vector
def random_scale_batch(scale_axis, min_scale, max_scale, batch_size):
	scale_list = [random_scale(scale_axis, min_scale, max_scale) for i in range(batch_size)]
	return torch.stack(scale_list, dim=0)


# Generate random noise given a tensor and noise variance.
def __random_noise(tensor, noise_variance):
	noise_std = math.sqrt(noise_variance)
	return torch.randn(tensor.size(), dtype=tensor.dtype, device=tensor.device) * noise_std


# Augment a single SDF sample
def augment_sample(points, distances, args):
	augmented_points, augmented_distances = points, distances
	noise_std = math.sqrt(args.noise_variance)

	# Add noise to the points and distances
	if not args.no_noise:
		augmented_points += __random_noise(points_noise, args.noise_variance)
		augmented_distances += __random_noise(distances_noise, args.noise_variance)

	# Rotate
	if not args.no_rotation:
		rotation_quat = random_rotation(args.rotate_axis).to(augmented_points.device)
		augmented_points = rotate_point_cloud(augmented_points, rotation_quat)

	return (augmented_points, augmented_distances)


# Augment a batch of SDF samples
def augment_sample_batch(batch_points, batch_distances, args):
	augmented_points = augment_sample_batch_points(batch_points, args)
	augmented_distances = augment_sample_batch_distances(batch_distances, args)
	return (augmented_points, augmented_distances)


# Augment a batch of point samples
def augment_sample_batch_points(batch_points, args):
	batch_size = batch_points.size()[0]
	augmented_points = batch_points

	# Add noise to the points and distances
	if not args.no_noise:
		augmented_points += __random_noise(batch_points, args.noise_variance)

	# Rotate
	if not args.no_rotation:
		rotation_quat = random_rotation_batch(args.rotate_axis, batch_size).to(augmented_points.device)
		augmented_points = rotate_point_cloud_batch(augmented_points, rotation_quat)

	return augmented_points


# Augment a batch of SDF distances
def augment_sample_batch_distances(batch_distances, args):
	augmented_distances = batch_distances

	# Add noise to the points and distances
	if not args.no_noise:
		augmented_distances += __random_noise(batch_distances, args.noise_variance)

	return augmented_distances
