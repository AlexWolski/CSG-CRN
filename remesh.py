import argparse
import trimesh
import point_cloud_utils as pcu
from os import cpu_count
from multiprocessing.pool import ThreadPool
from utilities.file_utils import create_output_dir, create_output_subdir, get_mesh_files


# Subdirectory names
UNIFORM_FOLDER = 'uniform'
SURFACE_FOLDER = 'surface'
NEAR_SURFACE_FOLDER = 'near-surface'


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
	data_group.add_argument('--output_dir', type=str, default='./data/output', help='Output directory to store manifold meshes')
	data_group.add_argument('--resolution', type=int, default=20000, help='Resolution for the marching cubes algorithm (target number of leaf nodes in the octree)')
	data_group.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')


	# Parse and handle Help argument
	args, remaining_args = help_parser.parse_known_args()

	if args.help or not remaining_args:
		print()
		data_parser.print_help()
		exit()

	# Parse data settings
	data_parser.parse_args(args=remaining_args, namespace=args)

	return args


def mesh_data_to_manifold(vertices, faces, resolution):
	"""
	Extract a manifold surface from a list of vertices and faces.

	Parameters
	----------
	vertices : numpy.ndarray
		Numpy array of shape (N, 3) where N is the number of vertices.
	faces : numpy.ndarray
		Numpy array of shape (N, 3) where N is the number of faces.
	resolution : int
		Resolution for the marching cubes algorithm (target number of leaf nodes in the octree).

	Returns
	-------
	Tuple[numpy.ndarray, numpy.ndarray]
		Two numpy arrays containing the vertices and faces of the resultant manifold mesh.

	"""
	return pcu.make_mesh_watertight(vertices, faces, resolution)


def mesh_to_manifold(mesh, resolution):
	"""
	Extract a manifold surface from a given mesh.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to convert to a manifold mesh.
	resolution : int
		Resolution for the marching cubes algorithm (target number of leaf nodes in the octree).

	Returns
	-------
	trimesh.Trimesh
		Resultant manifold mesh.

	"""
	vertices, faces = mesh_data_to_manifold(mesh.vertices, mesh.faces, resolution)
	return trimesh.Trimesh(vertices=vertices, faces=faces)


def file_to_manifold(data_dir, output_dir, mesh_path, resolution):
	"""
	Extract a manifold surface from a given mesh.

	Parameters
	----------
	data_dir : str
		Parent directory containing input 3D data files (3mf, obj, off, glb, gltf, ply, stl, 3dxml).
	output_dir : str
		Output directory to store manifold meshes.
	mesh_path : str
		Model file path to convert to a manifold.
	resolution : int
		Resolution for the marching cubes algorithm (target number of leaf nodes in the octree).

	"""
	mesh = trimesh.util.concatenate(trimesh.load(mesh_path).dump())
	manifold_path = create_output_subdir(data_dir, output_dir, mesh_path)
	manifold_mesh = mesh_to_manifold(mesh, resolution)
	file_extention = manifold_path.split('.')[-1]
	trimesh.exchange.export.export_mesh(manifold_mesh, manifold_path, file_extention)


def process_dataset(data_dir, output_dir, resolution, overwrite):
	"""
	Create output directory

	Parameters
	----------
	data_dir : str
		Parent directory containing input 3D data files (3mf, obj, off, glb, gltf, ply, stl, 3dxml).
	output_dir : str
		Output directory to store manifold meshes.
	resolution : int
		Resolution for the marching cubes algorithm (target number of leaf nodes in the octree).
	overwrite : bool
		Overwrite existing files in output directory.

	"""
	# Get all mesh files in the data directory
	mesh_paths = get_mesh_files(args.data_dir)

	# Check that the data directory contains mesh files
	if len(mesh_paths) == 0:
		print(f'No 3D data files found in directory "{data_dir}.\nUse one of the following data types: (3mf, obj, off, glb, gltf, ply, stl, 3dxml)"')
		return

	create_output_dir(output_dir, overwrite)
	print(f'Processing {len(mesh_paths)} files...')

	# Convert a mesh file to manifold and save to file on each CPU core
	with ThreadPool(cpu_count()) as pool:
		file_to_manifold_func = lambda mesh_path: file_to_manifold(data_dir, output_dir, mesh_path, resolution)
		pool.map_async(file_to_manifold_func, mesh_paths, chunksize=1).get()


if __name__ == '__main__':
	args = options()
	process_dataset(args.data_dir, args.output_dir, args.resolution, args.overwrite)
