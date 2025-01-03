import os
import glob
import argparse
import numpy as np
import torch
import trimesh
import mesh_to_sdf
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from mesh_to_sdf.utils import scale_to_unit_sphere
from mesh_to_sdf import BadMeshException
from tqdm import tqdm
from utilities import data_augmentation


# Parse commandline arguments
def options():
	# Parsers
	help_parser = argparse.ArgumentParser(add_help=False)
	data_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	augment_parser = data_augmentation.get_augment_parser('OFFLINE AUGMENT SETTINGS')
	data_group = data_parser.add_argument_group('DATA SETTINGS')

	# Help flag
	help_parser.add_argument('-h', '--help', default=False, action='store_true', help='Print help text')

	# Data settings
	data_group.add_argument('--data_dir', type=str, required=True, help='Parent directory containing input 3D data files (3mf, obj, off, glb, gltf, ply, stl, 3dxml)')
	data_group.add_argument('--output_dir', type=str, default='./data/output', help='Output directory to store SDF samples')
	data_group.add_argument('--num_sample_points', type=int, default=10000, help='Number of points to sample for each SDF')
	data_group.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')


	# Parse and handle Help argument
	args, remaining_args = help_parser.parse_known_args()

	if args.help or not remaining_args:
		print()
		data_parser.print_help()
		print('\n')
		augment_parser.print_help()
		exit()

	# Parse data settings
	args, remaining_args = data_parser.parse_known_args(args=remaining_args, namespace=args)

	# Parse augment settings
	augment_parser.parse_args(args=remaining_args, namespace=args)

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


# Create subdirectory for a given model file and return the path
def create_output_subdir(data_dir, output_dir, mesh_file_path):
	# Get path to input 3D model file
	rel_path = os.path.relpath(mesh_file_path, data_dir)
	model_path = os.path.join(output_dir, rel_path)

	# Create any subdirectories if they don't exist
	output_subdir = os.path.dirname(os.path.realpath(model_path))
	os.makedirs(output_subdir, exist_ok=True)

	return model_path


# Compute SDF samples
def sample_sdf(mesh_file_path, num_sample_points):
	# Prepare mesh
	mesh = trimesh.load(mesh_file_path)
	mesh = scale_to_unit_sphere(mesh)

	# Sample mesh surface
	points = sample_uniform_points_in_unit_sphere(num_sample_points).astype(np.float32)
	distances = mesh_to_sdf.mesh_to_sdf(mesh, points).astype(np.float32)

	# Convert to torch tensor
	points = torch.from_numpy(points)
	distances = torch.from_numpy(distances)

	return (points, distances)


# Save sample to .npy file
def save_sample(points, distances, output_path):
	sample = torch.cat((points, distances.unsqueeze(0).transpose(0, 1)), dim=1)
	np.save(output_path, sample)


# Save augment copies
def save_augment_samples(model_path, points, distances, args):
	# Augment and save samples
	augment_count = 0

	for i in range(args.augment_copies):
		# Adjsut file name for duplicate samples
		model_path = os.path.splitext(model_path)[0]
		output_path = model_path

		if augment_count > 0:
			output_path = output_path + f' ({augment_count})'

		# Augment sample
		(augmented_points, augmented_distances) = data_augmentation.augment_sample(points, distances, args)

		# Save samples to .npy file
		save_sample(augmented_points, augmented_distances, output_path)
		augment_count += 1


# Compute SDF samples for all 3D files in given directory
def prepare_dataset(data_dir, output_dir, args):
	# Convert all meshes
	mesh_file_paths = get_mesh_files(args.data_dir)
	create_output_dir(args.output_dir)
	print(f'Processing {len(mesh_file_paths)} files...')

	for mesh_file_path in tqdm(mesh_file_paths):
		# Compute SDF samples
		try:
			(points, distances) = sample_sdf(mesh_file_path, args.num_sample_points)
		# Skip meshes that raise an error
		except BadMeshException:
			tqdm.write(f'Skipping Bad Mesh\n: {mesh_file_path}')
			continue

		# Create model output path
		model_path = create_output_subdir(data_dir, output_dir, mesh_file_path)

		# Augment samples
		if args.augment_data:
			save_augment_samples(model_path, points, distances, args)
		else:
			save_sample(points, distances, output_path)

	print(f'Processing complete! Dataset saved to:\n{os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
	args = options()
	prepare_dataset(args.data_dir, args.output_dir, args)