import torch
from chamferdist import ChamferDistance
from utilities.csg_to_mesh import csg_to_mesh
from utilities.sampler_utils import sample_points_mesh_surface


def compute_chamfer_distance(target_surface_samples, recon_surface_samples):
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
	with torch.no_grad():
		chamferDist = ChamferDistance().to(target_surface_samples.device)
		dist_bidirectional = chamferDist(target_surface_samples, recon_surface_samples, bidirectional=True)
		return dist_bidirectional.detach().cpu().item()


def compute_chamfer_distance_csg(target_surface_samples, csg_model, num_acc_points, recon_resolution):
	"""
	Compute the Chamfer Distance metric between a target point cloud and a CSG reconstruction.
	Uses the marching cubes algorithm to extract an isosurface mesh of the CSG model and sample surface points.

	Parameters
	----------
	target_surface_samples : torch.Tensor
		Tensor of size (B, N, 3) containing B batches of target shapes represented by N surface points each.
	csg_model : utilities.csg_model.CSGModel
		CSG reconstruction model of a target shape.
	num_acc_points : int
		Number of points to use when computing Chamfer distance.

	Returns
	-------
	float
		The average bidirectional Chamfer Distance accuracy metric between all batches of target and reconstruction shapes.

	"""
	# Extract meshes from CSG models
	recon_mesh_list = csg_to_mesh(csg_model, recon_resolution)
	recon_points_list = []

	# Sample point clouds from meshes
	for recon_mesh in recon_mesh_list:
		recon_points = sample_points_mesh_surface(recon_mesh, num_acc_points).to(csg_model.device)
		recon_points_list.append(recon_points)

	# Compute average Chamfer distance
	recon_points_batch = torch.stack(recon_points_list)
	return compute_chamfer_distance(target_surface_samples, recon_points_batch)
