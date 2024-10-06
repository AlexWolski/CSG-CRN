import argparse
import numpy as np
import random
import torch
import math
from enum import Enum
from utilities.point_transform import rotate_point_cloud, scale_point_cloud
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


def get_augment_parser(group_name='AUGMENT SETTINGS'):
	parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	parser_group = parser.add_argument_group(group_name)

	parser_group.add_argument('--augment_data', default=False, action='store_true', help='Enable augmentation of object samples with random rotation, scaling, and noise')
	parser_group.add_argument('--augment_copies', type=int, default=1, help='Number of augmented copies of each object to create')
	parser_group.add_argument('--keep_original', default=False, action='store_true', help='Include the non-augmented object in an augmented output')
	parser_group.add_argument('--no_rotation', default=False, action='store_true', help='Disable rotations in data augmentation')
	parser_group.add_argument('--no_scale', default=False, action='store_true', help='Disable scaling in data augmentation')
	parser_group.add_argument('--no_noise', default=False, action='store_true', help='Disable gaussian noise for sample points and distances')
	parser_group.add_argument('--rotate_axis', default='ALL', type=RotationAxis, choices=list(RotationAxis), help='Axis to rotate around')
	parser_group.add_argument('--scale_axis', default='ALL', type=ScaleAxis, choices=list(ScaleAxis), help='Axes to scale')
	parser_group.add_argument('--min_scale', type=float, default=0.5, help='Lower bound on random scale value')
	parser_group.add_argument('--max_scale', type=float, default=2.0, help='Upper bound on random scale value')
	parser_group.add_argument('--noise_variance', type=float, default=1e-6, help='The variance of the gaussian noise aded to each sample point and distance')
	parser_group.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')

	return parser


# Convert a SciPy Rotation object to a quaternion PyTorch tensor of format [w, x, y, z]
def rotation_to_quat_tensor(rotation):
	return torch.from_numpy(rotation.as_quat(scalar_first=True).astype(np.float32))


# Generate a random rotation quaternion
def random_rotation(args):
	rotate_axis = args.rotate_axis

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


# Generate a random scale vector
def random_scale(args):
	scale_axis = args.scale_axis

	# Generate a random value for each axis
	if scale_axis == ScaleAxis('ALL'):
		scale_vec = [random.uniform(args.min_scale, args.max_scale) for i in range(3)]
		return torch.FloatTensor(scale_vec)

	# Select an axis to scale
	if scale_axis == ScaleAxis('RANDOM'):
		scale_axis = ScaleAxis.random_value()

	# Generate a random scalar for each axis
	axes = [ScaleAxis('X'), ScaleAxis('Y'), ScaleAxis('Z')]
	scale_vec = [random.uniform(args.min_scale, args.max_scale)
				if scale_axis == axes[i] else 1.0 for i in range(3)]

	return torch.FloatTensor(scale_vec)


# Generate a list of augmented copies
def generate_augmented_copies(points, distances, args):
	augmented_samples_list = []

	# Save original points and distances
	if args.keep_original:
		augmented_samples_list.append((points, distances))

	# Create and save augmented copies
	for i in range(args.augment_copies):
		augmented_samples_list.append(augment_sample(points, distances, args))

	return augmented_samples_list


# Augment a single SDF sample
def augment_sample(points, distances, args):
	augmented_points, augmented_distances = points, distances
	noise_std = math.sqrt(args.noise_variance)

	# Rotate
	if not args.no_rotation:
		rotation_quat = random_rotation(args)
		augmented_points = rotate_point_cloud(augmented_points, rotation_quat)

	# Scale
	if not args.no_scale:
		scale_vec = random_scale(args)
		augmented_points = scale_point_cloud(augmented_points, scale_vec)

	# Add noise to the points and distances
	if not args.no_noise:
		points_noise = torch.randn(points.size(), dtype=points.dtype, device=points.device) * noise_std
		distances_noise = torch.randn(distances.size(), dtype=distances.dtype, device=distances.device) * noise_std
		augmented_points += points_noise
		augmented_distances += distances_noise

	return (augmented_points, augmented_distances)