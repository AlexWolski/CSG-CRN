import torch
from chamferdist import ChamferDistance
from utilities.csg_to_mesh import csg_to_mesh
from utilities.sampler_utils import sample_points_mesh_surface, sample_points_csg_surface, sample_sdf_near_csg_surface


def compute_chamfer_distance(target_surface_samples, recon_surface_samples, no_grad=False):
	"""
	Compute the Chamfer Distance metric between a target point cloud and a reconstruction point cloud.

	Parameters
	----------
	target_surface_samples : torch.Tensor
		Tensor of size (B, N, 3) containing B batches of target shapes represented by N surface points each.
	recon_surface_samples : torch.Tensor
		Tensor of size (B, N, 3) containing B batches of reconstructed shapes represented by N surface points each.

	Returns
	-------
	float
		The average bidirectional Chamfer Distance accuracy metric between all batches of target and reconstruction shapes.

	"""
	with torch.set_grad_enabled(not no_grad):
		chamferDist = ChamferDistance().to(target_surface_samples.device)
		dist_bidirectional = chamferDist(target_surface_samples, recon_surface_samples, bidirectional=True)

		if no_grad:
			return dist_bidirectional.detach().cpu().item()
		else:
			return dist_bidirectional.cpu()


def compute_chamfer_distance_csg(target_surface_samples, csg_model, num_acc_points, recon_resolution):
	"""
	Compute the Chamfer Distance metric between a target point cloud and a CSG reconstruction.
	Uses the marching cubes algorithm to extract an isosurface mesh of the CSG model and sample surface points.
	This method is non-differentiable as the marching cubes algorithm is non-differentiable.

	Parameters
	----------
	target_surface_samples : torch.Tensor
		Tensor of size (B, N, 3) containing B batches of target shapes represented by N surface points each.
	csg_model : utilities.csg_model.CSGModel
		CSG reconstruction model of a target shape.
	num_acc_points : int
		Number of points to use when computing Chamfer distance.
	recon_resolution : int
		Resolution for the marching cubes algorithm (target number of leaf nodes in the octree).

	Returns
	-------
	float
		The average bidirectional Chamfer Distance accuracy metric between all batches of target and reconstruction shapes.

	"""
	# Sample CSG surface
	recon_points_batch = sample_points_csg_surface(csg_model, recon_resolution, num_acc_points)
	# Compute average Chamfer distance
	return compute_chamfer_distance(target_surface_samples, recon_points_batch, no_grad=True)


def compute_chamfer_distance_csg_fast(target_surface_samples, csg_model, num_acc_points, sample_dist):
	"""
	Compute the approximate Chamfer Distance metric between a target point cloud and a CSG reconstruction.
	Uses a heuristic method to efficiently generate surface samples of a CSG model within a threshold distance `sample_dist`.

	Parameters
	----------
	target_surface_samples : torch.Tensor
		Tensor of size (B, N, 3) containing B batches of target shapes represented by N surface points each.
	csg_model : utilities.csg_model.CSGModel
		CSG reconstruction model of a target shape.
	num_acc_points : int
		Number of points to use when computing Chamfer distance.
	sample_dist : float
		Maximum distance between generated points and the corresponding CSG model isosurface.

	Returns
	-------
	float
		The average bidirectional Chamfer Distance accuracy metric between all batches of target and reconstruction shapes.

	"""
	# Sample CSG surface
	(recon_points_batch, _) = sample_sdf_near_csg_surface(csg_model, num_acc_points, sample_dist)
	# Compute average Chamfer distance
	return compute_chamfer_distance(target_surface_samples, recon_points_batch)
