import os
import glob
import argparse
import numpy as np
import trimesh
import mesh_to_sdf
from enum import Enum
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

    def __str__(self):
        return self.value

    # Make constructor case-insensitive
    @classmethod
    def _missing_(self, value):
        return self(value.upper())


class ScaleAxis(Enum):
    xAxis = 'X'
    yAxis = 'Y'
    zAxis = 'Z'
    xyPlane = 'XY'
    yzPlane = 'YZ'
    xzPlane = 'XZ'
    random = 'RANDOM'

    def __str__(self):
        return self.value

    # Make constructor case-insensitive
    @classmethod
    def _missing_(self, value):
        return self(value.upper())


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
	parser.add_argument('--mo_scale', default=False, action='store_true', help='Disable scaling in data augmentation')
	parser.add_argument('--rotate_axis', type=RotationAxis, choices=list(RotationAxis), help='Axis to rotate around')
	parser.add_argument('--scale_axis', type=ScaleAxis, choices=list(ScaleAxis), help='Axes to scale')
	parser.add_argument('--min_scale', type=float, default=1.0, help='Lower bound on random scale value')
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
		err_msg = f'The output folder "{output_dir}" is already populated'
		raise Exception(err_msg)


# Compute SDF samples 
def sample_sdf(mesh_file_path, num_samples):
	mesh = trimesh.load(mesh_file_path)
	mesh = scale_to_unit_sphere(mesh)

	points = sample_uniform_points_in_unit_sphere(num_samples).astype(np.float32)
	distance = mesh_to_sdf.mesh_to_sdf(mesh, points).astype(np.float32)
	sdf_samples = np.concatenate((points, np.expand_dims(distance, axis=1)), 1)

	return sdf_samples


# Compute SDF samples for all 3D files in given directory
def prepare_dataset(data_dir, output_dir, args):
	# Convert all meshes
	mesh_file_paths = get_mesh_files(args.data_dir)
	print(f'Processing {len(mesh_file_paths)} files...')

	for mesh_file_path in tqdm(mesh_file_paths):
		try:
			# Compute SDF samples
			sdf_samples = sample_sdf(mesh_file_path, args.num_samples)

			# Get output file path
			rel_path = os.path.relpath(mesh_file_path, data_dir)
			output_path = os.path.join(output_dir, rel_path)

			# Create any subdirectories if they don't exist
			output_subdir = os.path.dirname(os.path.realpath(output_path))
			os.makedirs(output_subdir, exist_ok=True)

			# Save samples to .npy file
			output_path = os.path.splitext(output_path)[0]
			np.save(output_path, sdf_samples)

		except BadMeshException:
			tqdm.write(f'Skipping Bad Mesh\n: {mesh_file_path}')


if __name__ == '__main__':
	args = options()

	create_output_dir(args.output_dir)
	prepare_dataset(args.data_dir, args.output_dir, args)