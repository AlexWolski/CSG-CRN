import torch
from chamferdist import ChamferDistance
from utilities.sampler_utils import sample_points_csg_surface, sample_sdf_near_csg_surface


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


########################################################################################################
## Earth Movers Distance Implementation from 3D Point Cloud Modeling Paper (2017, Achlioptas et. al.) ##
########################################################################################################


def approxMatch(X, Y):
    """
    Calculates Approximate Matching. An iterative algorithm to calculate 
    matching matrix.

    Parameter : 
    X (torch.tensor) : 3D point cloud of dimension (2048,3)
    Y (torch.tensor) : 3D point cloud of dimension (2048,3)

    Returns:
    (torch.tensor) : A matching matrix of dimension (2048, 2048) 
    """

    n = X.shape[0]
    m = Y.shape[0]
    factorl = max(n,m)/n
    factorr = max(n,m)/m

    device = X.get_device()

    saturatedl = torch.ones(n,dtype=torch.float).to(device)*factorl
    saturatedr = torch.ones(m,dtype=torch.float).to(device)*factorr

    match = torch.zeros((n,m),dtype=torch.float).to(device)

    for i in range(7,-3,-1):
        level = -torch.pow(torch.tensor(4.0),torch.tensor(i)).to(device)
        if i == -2:
            level = torch.tensor(0, dtype=torch.float).to(device)
    
        
        weight = torch.exp(level*torch.cdist(X, Y)*torch.cdist(X, Y))*saturatedr
        # print(weight)

        s = torch.sum(weight, axis=1)
        s[s < 1e-9] = 1e-9

        weight = weight/s[:,None]*saturatedl[:,None]
        ss = torch.ones(n,dtype=torch.float).to(device)*1e-9
        ss += torch.sum(weight, axis=0)
        ss[ss < 1e-9] = 1e-9

        ss = saturatedr/ss 
        ss[ss>1.0] = 1.0

        weight = weight*ss
        s = torch.sum(weight, axis=0)
        ss2 = torch.sum(weight, axis=0)

        saturatedl = saturatedl-s
        saturatedl[saturatedl < 0] = 0

        match = match + weight

        saturatedr = saturatedr - ss2
        saturatedr[saturatedr < 0] = 0

    return match


def matchCost(X, Y, match):
    """
    Calculates Loss

    Parameter : 
    X (torch.tensor) : 3D point cloud of dimension (2048,3)
    Y (torch.tensor) : 3D point cloud of dimension (2048,3)
    (torch.tensor) : A matching matrix of dimension (2048, 2048) from
                    approxMatch function.

    Returns:
    (int) : Earth Mover Distance 
    """ 
    
    cost = (torch.cdist(X, Y)*match).sum()
    return cost


def EMD(X,Y):
	"""
	Calculates the Earth Mover Distance between two point clouds 
	using Auction LAP

	d_{EMD}(S_1, S_2) = \min_{\phi: S_1 \rightarrow S_2} \sum_{x \in S_1} ||x-\phi(x)|| \quad \textrm{where} \ \phi : S_1 \rightarrow S_2 \ \textrm{is bijection}

	Step 1) Calculate the approximate match matrix between 3D point cloud
	Step 2) Calculate the cost function based on match matrix

	Parameter : 
	X (torch.tensor) : 3D point cloud with dimension (2048,3)
	Y (torch.tensor) : 3D point cloud with dimension (2048,3)

	Returns:
	(int) : Earth mover distance between 3D point clouds X and Y

	Reference:
	1) https://github.com/optas/latent_3d_points/blob/master/external/structural_losses/approxmatch.cpp
	2) https://github.com/daerduoCarey/PyTorchEMD
	
	"""
	mat = approxMatch(X, Y)
	cost = matchCost(X, Y, mat)

	return cost

	
def EMDBatch(X_batch, Y_batch):
	"""
	Earth Mover distance between two batches of pointcloud
	"""
	loss = [EMD(X_batch[i],Y_batch[i]) for i in range(X_batch.shape[0])]
	return loss
