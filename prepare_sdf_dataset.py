import argparse
import numpy as np
import os
import torch
import trimesh
import yaml
from mesh_to_sdf import BadMeshException
from mesh_to_sdf.utils import scale_to_unit_sphere
from tqdm import tqdm
from utilities.data_processing import UNIFORM_FOLDER, SURFACE_FOLDER, NEAR_SURFACE_FOLDER, SETTINGS_FILE, SAMPLE_LIST_FILE, save_list
from utilities.file_utils import create_output_dir, create_output_subdir, get_mesh_files
from utilities.sampler_utils import sample_from_mesh


# Parse command-line arguments
def options():
	# Parsers
	help_parser = argparse.ArgumentParser(add_help=False)
	data_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	data_group = data_parser.add_argument_group('DATA SETTINGS')

	# Help flag
	help_parser.add_argument('-h', '--help', default=False, action='store_true', help='Print help text')

	# Data settings
	data_group.add_argument('--data_dir', type=str, required=True, help='Parent directory containing input 3D data files (3mf, obj, off, glb, gltf, ply, stl, 3dxml)')
	data_group.add_argument('--output_dir', type=str, default='./data/output', help='Output directory to store SDF samples')
	data_group.add_argument('--num_sdf_samples', type=int, default=4096, help='Number of uniform and near-surface SDF samples to generate')
	data_group.add_argument('--num_surface_samples', type=int, default=30000, help='Number of surface samples to generate for accuracy computation')
	data_group.add_argument('--sample_dist', type=float, default=0.1, help='Maximum distance to object surface for near-surface sampling (must be > 0)')
	data_group.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')


	# Parse and handle Help argument
	help_arg, remaining_args = help_parser.parse_known_args()

	if help_arg.help or not remaining_args:
		print()
		data_parser.print_help()

		exit()

	# Parse data settings
	args, remaining_args = data_parser.parse_known_args(args=remaining_args)

	return args


# Create output directory
def create_output_dir(output_dir, overwrite=False):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	elif len(os.listdir(output_dir)) != 0 and not overwrite:
		err_msg = f'The output folder "{output_dir}" is already populated. Use another directory or the --overwrite option.'
		raise Exception(err_msg)


# Create dataset directories and metadata file
def init_dataset(args):
	uniform_dir = os.path.join(args.output_dir, UNIFORM_FOLDER)
	surface_dir = os.path.join(args.output_dir, SURFACE_FOLDER)
	near_surface_dir = os.path.join(args.output_dir, NEAR_SURFACE_FOLDER)
	metadata_path = os.path.join(args.output_dir, SETTINGS_FILE)

	create_output_dir(args.output_dir, args.overwrite)
	create_output_dir(uniform_dir, args.overwrite)
	create_output_dir(surface_dir, args.overwrite)
	create_output_dir(near_surface_dir, args.overwrite)

	with open(metadata_path, 'w') as out_path:
		yaml.dump(args.__dict__, out_path, sort_keys=False)

	return (uniform_dir, surface_dir, near_surface_dir)


# Save sample to .npy file
def save_sample(output_path, points, distances=None):
	output_path = os.path.splitext(output_path)[0]

	if distances != None:
		sample = torch.cat((points, distances.unsqueeze(0).transpose(0, 1)), dim=1)
	else:
		sample = points

	np.save(output_path, sample)


# Compute uniform, near-surface, and surface SDF samples of a mesh
def prepare_mesh(mesh_file_path, uniform_dir, surface_dir, near_surface_dir, args):
	mesh = trimesh.load(mesh_file_path)
	mesh = scale_to_unit_sphere(mesh)

	# Compute samples
	(
		uniform_points, uniform_distances,
		near_surface_points, near_surface_distances,
		surface_points
	) = sample_from_mesh(mesh, args.num_sdf_samples, args.num_sdf_samples, args.num_sdf_samples, args.sample_dist)

	# Create output path for each model
	uniform_path = create_output_subdir(args.data_dir, uniform_dir, mesh_file_path)
	near_surface_path = create_output_subdir(args.data_dir, near_surface_dir, mesh_file_path)
	surface_path = create_output_subdir(args.data_dir, surface_dir, mesh_file_path)

	# Save samples
	save_sample(uniform_path, uniform_points, uniform_distances)
	save_sample(near_surface_path, near_surface_points, near_surface_distances)
	save_sample(surface_path, surface_points)


# Compute SDF samples for all 3D files in given directory
def prepare_dataset(args):
	# Convert all meshes
	mesh_file_paths = get_mesh_files(args.data_dir)
	sample_paths = []
	(uniform_dir, surface_dir, near_surface_dir) = init_dataset(args)
	print(f'Processing {len(mesh_file_paths)} files...')

	for mesh_file_path in tqdm(mesh_file_paths):
		# Compute SDF samples
		try:
			# Prepare mesh
			mesh = trimesh.load(mesh_file_path)
			mesh = scale_to_unit_sphere(mesh)
			prepare_mesh(mesh_file_path, uniform_dir, surface_dir, near_surface_dir, args)

			# Save path to sample
			rel_path = os.path.relpath(mesh_file_path, args.data_dir)
			rel_path = os.path.splitext(rel_path)[0] + '.npy'
			sample_paths.append(rel_path)
		# Skip meshes that raise an error
		except BadMeshException:
			tqdm.write(f'Skipping Bad Mesh\n: {mesh_file_path}')
			continue

	# Save paths for each sample in the dataset
	file_list_path = os.path.join(args.output_dir, SAMPLE_LIST_FILE)
	save_list(file_list_path, sample_paths)

	print(f'Processing complete! Dataset saved to:\n{os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
	args = options()
	prepare_dataset(args)
