import os
import glob
import random
import argparse
import numpy as np
import torch
import trimesh
import mesh_to_sdf
from enum import Enum
from utilities.point_transform import rotate_point_cloud
from scipy.spatial.transform import Rotation
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from mesh_to_sdf.utils import scale_to_unit_sphere
from mesh_to_sdf import BadMeshException
from tqdm import tqdm


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


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', type=str, required=True, help='Parent directory containing input 3D data files (3mf, obj, off, glb, gltf, ply, stl, 3dxml)')
	parser.add_argument('--output_dir', type=str, default='./output', help='Output directory to store SDF samples')
	parser.add_argument('--num_samples', type=int, default=200000, help='Number of SDF samples to compute')
	parser.add_argument('--augment_data', default=False, action='store_true', help='Enable random rotation and scaling of object samples')
	parser.add_argument('--augment_copies', type=int, default=1, help='Number of augmented copies of each object to create')
	parser.add_argument('--keep_original', default=False, action='store_true', help='Include the non-augmented object in an augmented output')
	parser.add_argument('--no_rotation', default=False, action='store_true', help='Disable rotations in data augmentation')
	parser.add_argument('--no_scale', default=False, action='store_true', help='Disable scaling in data augmentation')
	parser.add_argument('--rotate_axis', default='ALL', type=RotationAxis, choices=list(RotationAxis), help='Axis to rotate around')
	parser.add_argument('--scale_axis', default='ALL', type=ScaleAxis, choices=list(ScaleAxis), help='Axes to scale')
	parser.add_argument('--min_scale', type=float, default=0.5, help='Lower bound on random scale value')
	parser.add_argument('--max_scale', type=float, default=2.0, help='Upper bound on random scale value')
	parser.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')

	args = parser.parse_args()
	return args


# Find all 3D mesh files in a given directory
def get_mesh_files(data_dir):
	mesh_file_types = ['*.3mf', '*.obj', '*.off', '*.glb', '*.gltf', '*.ply', '*.stl', '*.3dxml']
	mesh_file_paths = []

	# Find all 3D files in parent directory
	for mesh_file_type in mesh_file_types:
		mesh_file_paths.extend(glob.glob(os.path.join(data_dir, '**', mesh_file_type), recursive=True))

	if len(mesh_file_paths) == 0:
		err_msg = f'No 3D data files found in directory "{data_dir}.\nUse one of the following data types: (3mf, obj, off, glb, gltf, ply, stl, 3dxml)"'
		raise Exception(err_msg)

	return mesh_file_paths


# Create output directory
def create_output_dir(output_dir):
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	elif len(os.listdir(args.output_dir)) != 0 and not args.overwrite:
		err_msg = f'The output folder "{output_dir}" is already populated. Use another directory or the --overwrite option.'
		raise Exception(err_msg)


# Compute SDF samples
def sample_sdf(mesh_file_path, num_samples):
	# Prepare mesh
	mesh = trimesh.load(mesh_file_path)
	mesh = scale_to_unit_sphere(mesh)

	# Sample mesh surface
	points = sample_uniform_points_in_unit_sphere(num_samples).astype(np.float32)
	distance = mesh_to_sdf.mesh_to_sdf(mesh, points).astype(np.float32)
	sdf_samples = np.concatenate((points, np.expand_dims(distance, axis=1)), 1)

	# Convert to torch tensor
	sdf_samples = torch.from_numpy(sdf_samples)

	return sdf_samples


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


# Compute SDF samples for all 3D files in given directory
def prepare_dataset(data_dir, output_dir, args):
	# Convert all meshes
	mesh_file_paths = get_mesh_files(args.data_dir)
	create_output_dir(args.output_dir)
	print(f'Processing {len(mesh_file_paths)} files...')

	for mesh_file_path in tqdm(mesh_file_paths):
		# Compute SDF samples
		try:
			sdf_samples = sample_sdf(mesh_file_path, args.num_samples)
		# Skip meshes that raise an error
		except BadMeshException:
			tqdm.write(f'Skipping Bad Mesh\n: {mesh_file_path}')
			continue

		# Augment samples
		if args.augment_data:
			augmented_samples = augment_samples(sdf_samples, args)
		else:
			augmented_samples = [sdf_samples]

		# Save samples
		i = 0

		for augmented_sample in augmented_samples:
			# Get path to input 3D model file
			rel_path = os.path.relpath(mesh_file_path, data_dir)
			model_path = os.path.join(output_dir, rel_path)

			# Create any subdirectories if they don't exist
			output_subdir = os.path.dirname(os.path.realpath(model_path))
			os.makedirs(output_subdir, exist_ok=True)

			# Adjsut file name for duplicate samples
			model_path = os.path.splitext(model_path)[0]
			output_path = model_path

			if i > 0:
				output_path = output_path + f' ({i})'

			# Save samples to .npy file
			np.save(output_path, augmented_sample.numpy())
			i += 1


if __name__ == '__main__':
	args = options()
	prepare_dataset(args.data_dir, args.output_dir, args)