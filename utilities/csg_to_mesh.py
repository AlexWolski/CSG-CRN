import torch
import trimesh
from skimage import measure
from utilities.csg_model import MIN_BOUND, MAX_BOUND


def get_grid_points(min_bound, max_bound, resolution, device=None):
	"""
	Generates a (N,N,N,3) grid of points where each point represents the position of a voxel.

	Parameters
	----------
	min_bound : float
		Minimum value along each axis of the grid.
	max_bound : float
		Maximum value along each axis of the grid.
	resolution : int
		Number of voxels to create along each axis. Grid contains `resolution`^3 total voxels.
	device : string, optional
		String representing device to create new grid tensor on. Default is None.

	Returns
	-------
	torch.Tensor
		Tensor of size (N, N, N, 3) where N=`resolution`.
		Represents a voxel grid where each of the N^3 points represents the position of a voxel.

	"""
	axis_values = torch.linspace(min_bound, max_bound, resolution, device=device)
	grid_axes = torch.meshgrid(axis_values, axis_values, axis_values, indexing='ij')
	grid_points = torch.stack(grid_axes, dim=-1)
	return grid_points.detach()


def csg_to_mesh(csg_model, resolution, iso_level=0.0):
	"""
	Use the marching cubes algorithm to extract a mesh representation of a given implicit CSG model.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		CSG model object to convert to a mesh.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	iso_level : float, optional
		The distance from the  isosurface. Default is 0.

	Returns
	-------
	trimesh.Trimesh
		Converted mesh object.

	"""
	voxel_size = abs(MAX_BOUND - MIN_BOUND) / resolution
	grid_points = get_grid_points(MIN_BOUND, MAX_BOUND, resolution, csg_model.device)

	# Reshape to (B, N, 3) where B=batch_size and N=num_points
	flat_points = grid_points.reshape(1, -1, 3)
	flat_distances = csg_model.sample_csg(flat_points)
	grid_distances = flat_distances.reshape(resolution, resolution, resolution)

	# Send distances tensor to CPU and convert to numpy
	grid_distances = grid_distances.cpu().detach().numpy()
	verts, faces, normals, values = measure.marching_cubes(grid_distances, level=0.0, spacing=[voxel_size] * 3)

	# Transform vertices from voxel space to object space
	offset = abs(MAX_BOUND - MIN_BOUND) / 2.0
	verts = verts - offset

	# Generate mesh
	mesh = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
	return mesh


def sample_csg_surface(csg_model, resolution, num_samples):
	"""
	Uniformly sample points on the surface of an implicit CSG model.
	Uses the marching cubes algorithm to extract an isosurface mesh, then uniformly samples the mesh faces.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	num_samples: int
		Number of surface samples to generate.

	Returns
	-------
	torch.Tensor
		Tensor of size (N, 3) where N=`num_samples`.
		Each point in the tensor is approximately on the surface of the given CSG model.

	"""
	mesh = csg_to_mesh(csg_model, resolution)
	samples = trimesh.sample_surface(mesh, num_samples)
	return torch.from_numpy(samples).detach()
