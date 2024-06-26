import os
import glob
import argparse
import math
import numpy as np
import torch
import trimesh
import mesh_to_sdf
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from mesh_to_sdf.utils import scale_to_unit_sphere
from mesh_to_sdf import BadMeshException
from tqdm import tqdm
from utilities.data_augmentation import RotationAxis, ScaleAxis, augment_samples


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', type=str, required=True, help='Parent directory containing input 3D data files (3mf, obj, off, glb, gltf, ply, stl, 3dxml)')
	parser.add_argument('--output_dir', type=str, default='./data/output', help='Output directory to store SDF samples')
	parser.add_argument('--num_samples', type=int, default=200000, help='Number of SDF samples to compute')
	parser.add_argument('--augment_data', default=False, action='store_true', help='Enable offline augmentation of object samples with random rotation, scaling, and noise')
	parser.add_argument('--augment_copies', type=int, default=1, help='Number of augmented copies of each object to create')
	parser.add_argument('--keep_original', default=False, action='store_true', help='Include the non-augmented object in an augmented output')
	parser.add_argument('--no_rotation', default=False, action='store_true', help='Disable rotations in data augmentation')
	parser.add_argument('--no_scale', default=False, action='store_true', help='Disable scaling in data augmentation')
	parser.add_argument('--no_noise', default=False, action='store_true', help='Disable gaussian noise for sample points and distances')
	parser.add_argument('--rotate_axis', default='ALL', type=RotationAxis, choices=list(RotationAxis), help='Axis to rotate around')
	parser.add_argument('--scale_axis', default='ALL', type=ScaleAxis, choices=list(ScaleAxis), help='Axes to scale')
	parser.add_argument('--min_scale', type=float, default=0.5, help='Lower bound on random scale value')
	parser.add_argument('--max_scale', type=float, default=2.0, help='Upper bound on random scale value')
	parser.add_argument('--noise_variance', type=float, default=1e-6, help='The variance of the gaussian noise aded to each sample point and distance')
	parser.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')

	args = parser.parse_args()
	args.noise_std = math.sqrt(args.noise_variance)

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

		print(f'Processing complete! Dataset saved to:\n{os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
	args = options()
	prepare_dataset(args.data_dir, args.output_dir, args)