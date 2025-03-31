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
		Float Tensor of size (N, 3) where N=`num_points`.
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
		Tensor of size (N, 3) where N=`num_points`.
		Points uniformly sampled form a sphere with radius `radius`.

	"""
	sphere_point_list = []
	num_sphere_points = 0
	total_points = num_points

	if batch_size != None:
		total_points *= batch_size

	while num_sphere_points < total_points:
		# Generate points uniformly distributed in a cube circumscribed on the unit sphere
		unit_cube_points = sample_uniform_points_cube(total_points*2, radius*2)
		# Select points encompassed by the unit sphere
		square_dist = unit_cube_points.square().sum(dim=-1)
		unit_sphere_points = unit_cube_points[square_dist <= 1]
		# Add selected points to results
		sphere_point_list.append(unit_sphere_points)
		num_sphere_points += unit_sphere_points.size(dim=0)

	# Select the needed number of point samples
	if sphere_point_list:
		unit_sphere_points = torch.cat(sphere_point_list, dim=0)[:total_points].detach()
	else:
		unit_sphere_points = torch.empty(0, 3)

	# Reshape if batch size is defined
	if batch_size != None:
		unit_sphere_points = torch.reshape(unit_sphere_points, (batch_size, num_points, 3))

	return unit_sphere_points


def distance_to_mesh_surface(mesh, sample_points):
	"""
	Compute the minimum signed distance between a tensor of points and a mesh.

	Parameters
	----------
	mesh : trimesh.Trimesh
		Mesh object to compute distances to.
	sample_points : torch.Tensor
		Tensor of size (N, 3) where N is the number of sample points.

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
		Tensor of size (N, 3) where N=`num_point_samples`.
		Points uniformly samples from the surface of the given mesh.

	"""
	if num_point_samples == 0:
		return torch.empty(0, 3)

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
		Two tensors of size (N, 3) and (N) where N=`num_sdf_samples`.
		SDF sample points and distances respectively.

	"""
	if num_sdf_samples == 0:
		return (torch.empty(0, 3), torch.empty(0))

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
		Two float tensors of size (N, 3) and (N) where N=`num_sdf_samples`.
		SDF sample points and distances respectively.

	"""
	if num_sdf_samples == 0:
		return (torch.empty(0, 3), torch.empty(0))

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


def sample_points_csg_surface(csg_model, resolution, num_sdf_samples):
	"""
	Uniformly sample points on the surface of a batch of CSG models.
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
	return torch.stack(surface_points_list).detach()


def sample_sdf_from_csg_uniform_sphere(csg_model, num_sdf_samples):
	"""
	Generate SDF samples a batch of CSG models uniformly distributed in a unit sphere.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	num_sdf_samples:  int
		Number of uniform SDF samples to generate.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor]
		Tensors of size (B, N, 3) and (B, N, 1) where B=`csg_model`.batch_size and N=`num_sdf_samples`.
		The first tensor contains points uniformly distributed in a unit sphere.
		The second tensor contains signed distances from each sample point to the CSG model isosurface.

	"""
	uniform_points = sample_uniform_points_sphere(num_sdf_samples, batch_size=csg_model.batch_size).to(csg_model.device)
	uniform_distances = csg_model.sample_csg(uniform_points)
	return (uniform_points, uniform_distances)


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
	Tuple[torch.Tensor, torch.Tensor]
		Tensors of size (B, N, 3) and (B, N, 1) where B=`csg_model`.batch_size and N=`num_sdf_samples`.
		The first tensor contains points within a distance `sample_dist` of the corresponding CSG model isosurface.
		The second tensor contains signed distances from each sample point to the CSG model isosurface.

	"""
	GEN_SAMPLE_MULTIPLE = 10
	radius = (MAX_BOUND - MIN_BOUND) / 2
	sample_points = sample_uniform_points_cube(num_sdf_samples, radius, csg_model.batch_size).to(csg_model.device)
	sample_distances = csg_model.sample_csg(sample_points)
	current_sample_dist = radius

	# Continuously generate new samples near the CSG model surface
	while current_sample_dist > sample_dist:
		# Generate new samples points by adding Gaussian noise
		new_points = sample_points.repeat(1, GEN_SAMPLE_MULTIPLE, 1)
		gaussian_noise = torch.randn(new_points.size(), dtype=new_points.dtype, device=new_points.device) * current_sample_dist
		new_points += gaussian_noise
		new_distances = csg_model.sample_csg(new_points)
		sample_points = torch.cat((sample_points, new_points), dim=1)
		sample_distances = torch.cat((sample_distances, new_distances), dim=1)

		# Keep `num_sdf_samples` samples with the lowest sample distance
		(new_sample_dist, sample_points, sample_distances) = _select_nearest_samples(sample_points, sample_distances, num_sdf_samples, current_sample_dist)

		# No samples could be found within the unit sphere
		if new_sample_dist == None:
			break

		current_sample_dist = new_sample_dist

	return (sample_points[:,:num_sdf_samples], sample_distances[:,:num_sdf_samples])


def _select_nearest_samples(batch_sample_points, batch_sample_distances, num_sdf_samples, current_sample_dist):
	"""
	Helper function to select `num_sdf_samples` samples with the smallest SDF distance.

	Parameters
	----------
	batch_sample_points : torch.Tensor
		Tensor of size (B, N, 1) containing point samples.
	batch_sample_distances : torch.Tensor
		Tensor of size (B, N, 1) containing SDF distance values.
	num_sdf_samples : int
		Required number of SDF samples.
	current_sample_dist : float
		Maximum value of SDF distances in `batch_sample_distances`.

	Returns
	-------
	Tuple[float, torch.Tensor, torch.Tensor]
		1. The maximum sample distance of the return SDF samples.
		2. Select point samples.
		3. Select SDF distance values.

	"""
	device = batch_sample_points.device
	batch_size = batch_sample_points.size(dim=0)

	NUM_BINS = 100
	bin_bounds = torch.linspace(0, current_sample_dist, steps=NUM_BINS+1, device=device)
	bin_indices = torch.bucketize(batch_sample_distances, bin_bounds)
	min_sample_dist = current_sample_dist
	min_bin_index = -1

	# Find minimum bin that contains at least num_sdf_samples samples
	for bin_index in range(0, NUM_BINS-1):
		bucket_count = torch.count_nonzero(bin_indices <= bin_index, dim=1)
		min_sample_count = torch.min(bucket_count).item()

		if min_sample_count >= num_sdf_samples:
			min_sample_dist = bin_bounds[bin_index+1]
			min_bin_index = bin_index
			break

	# No samples were found
	if min_bin_index == -1:
		return (None, batch_sample_points, batch_sample_distances)

	# Select num_sdf_samples samples
	batch_indices = bin_indices <= min_bin_index
	select_points_list = []
	select_distances_list = []

	for batch in range(batch_size):
		select_indices = batch_indices[batch]
		select_points = batch_sample_points[batch][select_indices]
		select_distances = batch_sample_distances[batch][select_indices]
		select_points_list.append(select_points[:num_sdf_samples])
		select_distances_list.append(select_distances[:num_sdf_samples])

	select_points = torch.stack(select_points_list)
	select_distances = torch.stack(select_distances_list)

	return (min_sample_dist, select_points, select_distances)


def sample_sdf_from_csg_combined(csg_model, num_sdf_samples, sample_dist, surface_uniform_ratio):
	"""
	Generate a combination of uniform and near-surface  SDF samples for a batch of CSG models.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	num_sdf_samples : int
		Number of surface samples to generate.
	surface_uniform_ratio : Percentage of near-surface samples to select.

	Returns
	-------
	Tuple[torch.Tensor, torch.Tensor]
		Two float tensors of size (B, N, 3) and (B, N, 1) where B=`csg_model`.batch_size and N=`num_sdf_samples`.

	"""
	# Generate uniform and surface samples
	num_uniform_samples = math.ceil(num_sdf_samples * surface_uniform_ratio)
	num_surface_samples = math.floor(num_sdf_samples * (1 - surface_uniform_ratio))
	(uniform_points, uniform_distances) = sample_sdf_from_csg_uniform_sphere(csg_model, num_uniform_samples)
	(surface_points, surface_distances) = sample_sdf_near_csg_surface(csg_model, num_surface_samples, sample_dist)

	# Combine samples
	combined_points = torch.cat((uniform_points, surface_points), dim=1)
	combined_distances = torch.cat((uniform_distances, surface_distances), dim=1)

	# Shuffle samples
	combined_points = combined_points[:, torch.randperm(num_sdf_samples)].detach()
	combined_distances = combined_distances[:, torch.randperm(num_sdf_samples)].detach()

	return (combined_points, combined_distances)
