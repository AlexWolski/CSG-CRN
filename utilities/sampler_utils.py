import mesh_to_sdf
import numpy
import point_cloud_utils as pcu
import trimesh
import torch
import math
from torch.distributions.uniform import Uniform
from utilities.csg_to_mesh import csg_to_mesh
from utilities.csg_model import MIN_BOUND, MAX_BOUND


def sample_uniform_points_cube(num_points, side_length, batch_size=None):
	"""
	Uniformly sample points from a cube with given side length.

	Parameters
	----------
	num_points : int
		Number of point samples to generate.
	side_length : float
		Side length of the cube to sample points from.
	batch_size : int
		Number of batches to generate

	Returns
	-------
	torch.Tensor
		Float Tensor of size (N, N, N) where N=`num_points`.
		Points uniformly sampled from a cube with side length `side_length`.

	"""
	low = torch.tensor(-side_length/2, dtype=torch.float32)
	high = torch.tensor(side_length/2, dtype=torch.float32)
	uniform_sampler = Uniform(low, high)

	if batch_size:
		return uniform_sampler.sample((batch_size, num_points, 3)).detach()
	else:
		return uniform_sampler.sample((num_points, 3)).detach()


def sample_uniform_points_sphere(num_points, radius=1, batch_size=None):
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
		unit_cube_points = sample_uniform_points_cube(num_points*2, radius*2, batch_size)
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
	Uniformly sample points on the surface of a batch of implicit CSG model.
	Uses the marching cubes algorithm to extract an isosurface mesh, then uniformly samples the mesh faces.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	num_sdf_samples : int
		Number of surface samples to generate.

	Returns
	-------
	torch.Tensor
		Tensor of size (B, N, 3) where B=`csg_model`.batch_size and N=`num_sdf_samples`.
		Each point in the tensor is on the surface of the given CSG model.

	"""
	# Extract meshes from CSG models
	mesh_list = csg_to_mesh(csg_model, resolution)
	surface_points_list = []

	# Sample point clouds from meshes
	for mesh in mesh_list:
		surface_points = sample_points_mesh_surface(mesh, num_sdf_samples).to(csg_model.device)
		surface_points_list.append(surface_points)

	# Compute average Chamfer distance
	return torch.stack(surface_points_list)


def sample_sdf_near_csg_surface(csg_model, num_sdf_samples, sample_dist):
	"""
	Generate SDF samples within a specified distance of the implicit surfaces of a batch of CSG models.
	Uses a recursive heuristic method of selecting samples closer to the implicit surface and adding Gaussian noise to generate new samples.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	num_sdf_samples:  int
		Number of surface samples to generate.
	sample_dist : float
		Maximum distance of each SDF sample to the surface of the corresponding CSG model surface.

	Returns
	-------
	torch.Tensor
		Tensors of size (B, N, 3) where B=`csg_model`.batch_size and N=`num_sdf_samples`.
		Each point in the tensor is within a distance `sample_dist` of the corresponding CSG model isosurface.

	"""
	side_length = MAX_BOUND - MIN_BOUND
	sample_points = sample_uniform_points_cube(num_sdf_samples, side_length, csg_model.batch_size).to(csg_model.device)
	sample_distances = csg_model.sample_csg(sample_points)
	initial_sample_dist = torch.mean(sample_distances).item()
	return _sample_sdf_near_csg_surface_helper(csg_model, num_sdf_samples, sample_dist, initial_sample_dist, sample_points, sample_distances)


# TODO: Convert this to non-recursive so that it uses less memory and we can create more random points
def _sample_sdf_near_csg_surface_helper(csg_model, num_sdf_samples, target_sample_dist, current_sample_dist, sample_points, sample_distances):
	"""
	Recursive helper function to generate SDF samples within a specified distance of the implicit surfaces of a batch of CSG models.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	num_sdf_samples : int
		Number of surface samples to generate.
	target_sample_dist : float
		Required sample distance for SDF samples.
	current_sample_dist : float
		Intermediate sample distance between `target_sample_dist` and `current_sample_dist`.
	sample_points : torch.Tensor
		Tensor of size (B, N, 3) containing N point samples with a maximum sample distance of `current_sample_dist`.
	sample_distances : torch.Tensor
		Tensor of size (B, N, 1) containing distances between point in `sample_points` and the corresponding CSG isosurface.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor]
		Two float tensors of size (B, N, 3) and (B, N, 1) where B=`csg_model`.batch_size and N=`num_sdf_samples`.
		SDF sample points and distances respectively.

	"""
	while True:
		# Base case is when the sample distance becomes sufficiently small
		if current_sample_dist <= target_sample_dist:
			return sample_points[:,:num_sdf_samples]

		# Reduce the sample distance
		current_sample_dist = max(target_sample_dist, current_sample_dist/5)
		(select_points, select_distances) = _select_samples_in_distance(csg_model, sample_points, sample_distances, current_sample_dist)

		# Generate new samples
		while select_distances.size(dim=1) < num_sdf_samples:
			SAMPLE_MULTIPLE = 10
			num_select_points = select_points.size(dim=1)
			num_to_gen = (num_sdf_samples * SAMPLE_MULTIPLE) - num_select_points
			multiple = math.ceil(num_to_gen / num_select_points)
			# Generate new samples points by adding Gaussian noise
			new_points = select_points.repeat(1, multiple, 1)
			gaussian_noise = torch.randn(new_points.size(), dtype=select_distances.dtype, device=csg_model.device) * current_sample_dist
			new_points += gaussian_noise
			new_distances = csg_model.sample_csg(new_points)

			# Select generated samples within the new sample distance
			(new_select_points, new_select_distances) = _select_samples_in_distance(csg_model, new_points, new_distances, current_sample_dist)
			select_points = torch.cat((select_points, new_select_points), dim=1)
			select_distances = torch.cat((select_distances, new_select_distances), dim=1)

		sample_points = select_points
		sample_distances = select_distances


def _select_samples_in_distance(csg_model, batch_sample_points, batch_sample_distances, min_dist):
	"""
	Helper function to select SDF samples within a specified distance threshold.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	batch_sample_points : torch.Tensor
		Tensor of size (B, N, 3) containing SDF sample points.
	batch_sample_distances : torch.Tensor
		Tensor of size (B, N, 1) containing SDf sample distances.
	min_dist : float
		Maximum allowed distance of SDF samples.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor]
		Two float tensors of size (B, N, 3) and (B, N, 1) where B=`csg_model`.batch_size
		and N is the number of SDF samples that have a distances less than or equal to `min_dist`.

	"""
	# Select SDF samples within the specified distance
	batch_select_indices = batch_sample_distances < min_dist

	# Find batch with the minimum number of select samples
	num_select_samples = torch.sum(batch_select_indices, dim=1)
	min_select_samples = torch.min(num_select_samples).item()

	# Combine select points and distances into tensor with min_select_samples samples
	select_points_list = []
	select_distances_list = []

	for batch in range(csg_model.batch_size):
		select_indices = batch_select_indices[batch]
		sample_points = batch_sample_points[batch]
		sample_distances = batch_sample_distances[batch]
		select_points = sample_points[select_indices]
		select_distances = sample_distances[select_indices]
		select_points_list.append(select_points[:min_select_samples])
		select_distances_list.append(select_distances[:min_select_samples])

	select_points = torch.stack(select_points_list)
	select_distances = torch.stack(select_distances_list)
	return (select_points, select_distances)
