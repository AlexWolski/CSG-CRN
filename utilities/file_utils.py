import os
import glob


def get_mesh_files(data_dir):
	"""
	Find all 3D mesh files in a given directory

	Parameters
	----------
	data_dir : str
		Parent directory containing input 3D data files (3mf, obj, off, glb, gltf, ply, stl, 3dxml).

	Returns
	-------
	list of str
		List of mesh file paths.

	"""
	mesh_file_types = ['*.3mf', '*.obj', '*.off', '*.glb', '*.gltf', '*.ply', '*.stl', '*.3dxml']
	mesh_file_paths = []

	# Find all 3D files in parent directory
	for mesh_file_type in mesh_file_types:
		mesh_file_paths.extend(glob.glob(os.path.join(data_dir, '**', mesh_file_type), recursive=True))

	return mesh_file_paths


def create_output_dir(output_dir, overwrite=False):
	"""
	Create output directory

	Parameters
	----------
	output_dir : str
		Output directory to store manifold meshes.
	overwrite : bool
		Overwrite existing files in output directory

	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	elif len(os.listdir(output_dir)) != 0 and not overwrite:
		err_msg = f'The output folder "{output_dir}" is already populated. Use another directory or the --overwrite option.'
		raise Exception(err_msg)


def create_output_subdir(source_dir, target_dir, file_path):
	"""
	Given a file path within a subdirectory of a source directory, create the same subdirectories in the output directory.
	Return a path to the given file but in the output directory.

	Parameters
	----------
	source_dir : str
		Parent directory containing `file_path`.
	target_dir : str
		Output directory where the new subdirectories are created.
	file_path : bool
		Path to a file within a subdirectory of `source_dir`.

	Returns
	-------
	str
		Path to the given file but in the output directory.

	"""
	# Get relative path between the source directory and the given file path 
	rel_path = os.path.relpath(file_path, source_dir)

	# Create any subdirectories if they don't exist
	output_file_path = os.path.join(target_dir, rel_path)
	output_subdir = os.path.dirname(os.path.realpath(output_file_path))
	os.makedirs(output_subdir, exist_ok=True)

	return output_file_path
