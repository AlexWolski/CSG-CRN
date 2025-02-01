import mesh_to_sdf
import point_cloud_utils as pcu
import trimesh
import torch
from torch.distributions.uniform import Uniform


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
		Tensor of size (N, N, N) where N=`num_points`.
		Points uniformly sampled from a cube with side length `side_length`.

	"""
	return Uniform(-side_length/2, side_length/2).sample((num_points, 3)).detach()


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
	sdf_points_numpy = sample_points.detach().numpy().astype(float)
	distances_numpy, _, _ = pcu.signed_distance_to_mesh(sdf_points_numpy, mesh.vertices, mesh.faces)
	return torch.from_numpy(distances_numpy)


def sample_points_mesh_surface(mesh, num_sample_points):
	"""
	Uniformly sample points from the surface of a mesh.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to sample from.
	num_sample_points : int
		Number of point samples to generate.

	Returns
	-------
	torch.Tensor
		Tensor of size (N, N, N) where N=`num_sample_points`.
		Points uniformly samples from the surface of the given mesh.

	"""
	(face_ids, barycentric_coords) = pcu.sample_mesh_random(mesh.vertices, mesh.faces, num_sample_points)
	surface_points = pcu.interpolate_barycentric_coords(mesh.faces, face_ids, barycentric_coords, mesh.vertices)
	return torch.from_numpy(surface_points)


def sample_sdf_near_surface(mesh, num_sample_points, sample_dist):
	"""
	Generate SDF samples of a mesh within a specified distance the mesh surface.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to sample from.
	num_sample_points : int
		Number of SDF samples to generate.
	sample_dist : float
		Maximum distance between each generated SDF sample and the mesh surface.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor]
		Two tensors of size (N, N, N) and (N) where N=`num_sample_points`.
		SDF sample points and distances respectively.

	"""
	surface_points = sample_points_mesh_surface(mesh, num_sample_points)
	gaussian_noise = torch.randn(surface_points.size(), dtype=surface_points.dtype, device=surface_points.device) * sample_dist
	near_surface_points = surface_points + gaussian_noise
	distances = distance_to_mesh_surface(mesh, near_surface_points)
	return (near_surface_points, distances)


def sample_sdf_unit_sphere(mesh, num_sample_points):
	"""
	Generate SDF samples of a mesh uniformly distributed in a unit sphere.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to sample from. The mesh is expected be scaled to a unit sphere.
	num_sample_points : int
		Number of SDF samples to generate.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor]
		Two tensors of size (N, N, N) and (N) where N=`num_sample_points`.
		SDF sample points and distances respectively.

	"""
	sample_points = sample_uniform_points_sphere(num_sample_points)
	sample_distances = distance_to_mesh_surface(mesh, sample_points)
	return (sample_points, sample_distances)
