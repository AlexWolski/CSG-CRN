import torch
from chamferdist import ChamferDistance


def compute_chamfer_distance(target_surface_samples, recon_surface_samples):
	"""
	Compute the Chamfer Distance metric between a target point cloud and a CSG reconstruction.
	Uses the marching cubes algorithm to extract an isosurface mesh of the CSG model and sample surface points.

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
