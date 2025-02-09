import mesh_to_sdf
import numpy
import point_cloud_utils as pcu
import trimesh
import torch
from torch.distributions.uniform import Uniform
from utilities.csg_to_mesh import csg_to_mesh


def sample_uniform_points_cube(num_points, side_length):
	"""
	Uniformly sample points from a cube with given side length.

	Parameters
	----------
	num_points : int
		Number of point samples to generate.
	side_length : float
		Side length of the cube to sample points from.

	Returns
	-------
	torch.Tensor
		Float Tensor of size (N, N, N) where N=`num_points`.
		Points uniformly sampled from a cube with side length `side_length`.

	"""
	low = torch.tensor(-side_length/2, dtype=torch.float32)
	high = torch.tensor(side_length/2, dtype=torch.float32)
	return Uniform(low, high).sample((num_points, 3)).detach()


def sample_uniform_points_sphere(num_points, radius=1):
	"""
	Uniformly sample points from a sphere with given radius.

	Parameters
	----------
	num_points : int
		Number of point samples to generate.
	radius : float
		Radius of the sphere to sample points from.

	Returns
	-------
	torch.Tensor
		Tensor of size (N, N, N) where N=`num_points`.
		Points uniformly sampled form a sphere with radius `radius`.

	"""
	sphere_point_list = []
	num_sphere_points = 0

	while num_sphere_points < num_points:
		# Generate points uniformly distributed in a cube circumscribed on the unit sphere
		unit_cube_points = sample_uniform_points_cube(num_points*2, radius*2)
		# Select points encompassed by the unit sphere
		square_dist = unit_cube_points.square().sum(dim=-1)
		unit_sphere_points = unit_cube_points[square_dist <= 1]
		# Add selected points to results
		sphere_point_list.append(unit_sphere_points)
		num_sphere_points += unit_sphere_points.size(dim=0)

	# Select the needed number of point samples
	unit_sphere_points = torch.cat(sphere_point_list, dim=0)
	return unit_sphere_points[:num_points].detach()


def distance_to_mesh_surface(mesh, sample_points):
	"""
	Compute the minimum signed distance between a tensor of points and a mesh.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to compute distances to.
	sample_points : torch.Tensor
		Tensor of size (N, N, N) where N is the number of sample points.

	Returns
	-------
	torch.Tensor
		Tensor of size (N) where N is the number of input points.
		Signed distance between each sample point and the closest point on the given mesh.

	"""
	sdf_points_numpy = sample_points.detach().numpy().astype(numpy.float32)
	distances_numpy, _, _ = pcu.signed_distance_to_mesh(sdf_points_numpy, mesh.vertices.astype(numpy.float32), mesh.faces)
	return torch.from_numpy(distances_numpy.astype(numpy.float32))


def sample_points_mesh_surface(mesh, num_point_samples):
	"""
	Uniformly sample points from the surface of a mesh.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to sample from.
	num_point_samples : int
		Number of point samples to generate.

	Returns
	-------
	torch.Tensor
		Tensor of size (N, N, N) where N=`num_point_samples`.
		Points uniformly samples from the surface of the given mesh.

	"""
	(face_ids, barycentric_coords) = pcu.sample_mesh_random(mesh.vertices, mesh.faces, num_point_samples)
	surface_points = pcu.interpolate_barycentric_coords(mesh.faces, face_ids, barycentric_coords, mesh.vertices)
	return torch.from_numpy(surface_points.astype(numpy.float32))


def sample_sdf_near_mesh_surface(mesh, num_sdf_samples, sample_dist):
	"""
	Generate SDF samples of a mesh within a specified distance the mesh surface.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to sample from.
	num_sdf_samples : int
		Number of SDF samples to generate.
	sample_dist : float
		Maximum distance between each generated SDF sample and the mesh surface.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor]
		Two tensors of size (N, N, N) and (N) where N=`num_sdf_samples`.
		SDF sample points and distances respectively.

	"""
	surface_points = sample_points_mesh_surface(mesh, num_sdf_samples)
	gaussian_noise = torch.randn(surface_points.size(), dtype=surface_points.dtype, device=surface_points.device) * sample_dist
	near_surface_points = surface_points + gaussian_noise
	distances = distance_to_mesh_surface(mesh, near_surface_points)
	return (near_surface_points, distances)


def sample_sdf_from_mesh_unit_sphere(mesh, num_sdf_samples):
	"""
	Generate SDF samples of a mesh uniformly distributed in a unit sphere.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to sample from. The mesh is expected be scaled to a unit sphere.
	num_sdf_samples : int
		Number of SDF samples to generate.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor]
		Two float tensors of size (N, N, N) and (N) where N=`num_sdf_samples`.
		SDF sample points and distances respectively.

	"""
	sample_points = sample_uniform_points_sphere(num_sdf_samples)
	sample_distances = distance_to_mesh_surface(mesh, sample_points)
	return (sample_points, sample_distances)


def sample_from_mesh(mesh, num_uniform_samples, num_surface_samples, num_near_surface_samples, sample_dist):
	"""
	Generate uniform SDF samples, near-surface SDF samples, and surface point samples of a given mesh.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to sample from. The mesh is expected be scaled to a unit sphere.
	num_uniform_samples : int
		Number of SDF samples to generate in a unit sphere around the mesh.
	num_surface_samples : int
		Number of point samples to generate on the mesh surface.
	num_near_surface_samples : int
		Number of SDF samples to generate within a maximum distance of the mesh surface.
	sample_dist : float
		Maximum distance between each generated SDF sample and the mesh surface.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
		Tuple of torch float tensors representing sampled points and distances:
		1. Uniform Points
		2. Uniform Distances
		3. Near-Surface Points
		4. Near-Surface Distances
		5. Surface Points

	"""
	(uniform_points, uniform_distances) = sample_sdf_from_mesh_unit_sphere(mesh, num_uniform_samples)
	(near_surface_points, near_surface_distances) = sample_sdf_near_mesh_surface(mesh, num_near_surface_samples, sample_dist)
	surface_points = sample_points_mesh_surface(mesh, num_surface_samples)

	return (
		uniform_points, uniform_distances,
		near_surface_points, near_surface_distances,
		surface_points
	)


def sample_csg_surface(csg_model, resolution, num_sdf_samples):
	"""
	Uniformly sample points on the surface of an implicit CSG model.
	Uses the marching cubes algorithm to extract an isosurface mesh, then uniformly samples the mesh faces.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	num_sdf_samples: int
		Number of surface samples to generate.

	Returns
	-------
	torch.Tensor
		Tensor of size (N, 3) where N=`num_sdf_samples`.
		Each point in the tensor is approximately on the surface of the given CSG model.

	"""
	mesh_list = csg_to_mesh(csg_model, resolution)
	# TODO: Convert to batch operation
	return sample_points_mesh_surface(mesh_list[0], num_sdf_samples).unsqueeze(0)


def sample_sdf_near_csg_surface(csg_model, resolution, num_sdf_samples, sample_dist):
	"""
	Generate SDF samples of a CSG within a specified distance the isosurface.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	num_sdf_samples: int
		Number of surface samples to generate.

	"""
	mesh_list = csg_to_mesh(csg_model, resolution)
	# TODO: Convert to batch operation
	return sample_sdf_near_mesh_surface(mesh_list[0], num_sdf_samples, sample_dist).unsqueeze(0)
