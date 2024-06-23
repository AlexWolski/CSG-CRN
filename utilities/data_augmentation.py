import numpy as np
import random
import torch
from enum import Enum
from utilities.point_transform import rotate_point_cloud
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
		return self(value.upper())

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
		return self(value.upper())

	def __str__(self):
		return self.value

	# Randomly select one of the 3 concrete enum states X, Y, and Z
	@classmethod
	def random_value(self):
		index = random.randrange(3)
		return list(ScaleAxis)[index]


# Convert a SciPy Rotation object to a quaternion PyTorch tensor
def rotation_to_quat_tensor(rotation):
	return torch.from_numpy(rotation.as_quat().astype(np.float32))


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
		return scale_vec

	# Select an axis to scale
	if scale_axis == ScaleAxis('RANDOM'):
		scale_axis = ScaleAxis.random_value()

	# Generate a random scalar for each axis
	axes = [ScaleAxis('X'), ScaleAxis('Y'), ScaleAxis('Z')]
	scale_vec = [random.uniform(args.min_scale, args.max_scale)
				if scale_axis == axes[i] else 1.0 for i in range(3)]

	return scale_vec


# Rotate and scale SDF point clouds
def augment_samples(sdf_samples, args):
	augmented_samples_list = []

	# Save original samples
	if args.keep_original:
		augmented_samples_list.append(sdf_samples)
	
	points = sdf_samples[:,:3]
	distances = sdf_samples[:,3]
	distances = distances.unsqueeze(0).transpose(0, 1)

	# Augment and save samples
	for i in range(args.augment_copies):
		augmented_points = points

		# Rotate
		if not args.no_rotation:
			rotation_quat = random_rotation(args)
			augmented_points = rotate_point_cloud(augmented_points, rotation_quat)

		# Scale
		if not args.no_scale:
			scale_vec = random_scale(args)
			# augmented_scale = scale_point_cloud(augmented_points, scale_vec)

		# Save augmented samples
		augmented_samples = torch.cat((augmented_points, distances), dim=1)
		augmented_samples_list.append(augmented_samples)

	return augmented_samples_list
