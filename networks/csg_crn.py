import torch
import torch.nn as nn
from networks.pointnet import PointNetfeat, POINTNET_FEAT_OUTPUT_SIZE
from networks.regressor_decoder import PrimitiveRegressor
from utilities.csg_model import CSGModel
from utilities.sampler_utils import sample_sdf_from_csg_combined


class CSG_CRN(nn.Module):
	def __init__(
			self, num_prims, num_shapes, num_operations, num_input_points, sample_dist, surface_uniform_ratio, device,
			decoder_layers=[], predict_blending=True, predict_roundness=True, no_batch_norm=False):
		super(CSG_CRN, self).__init__()

		self.num_prims = num_prims
		self.num_shapes = num_shapes
		self.num_operations = num_operations
		self.num_input_points = num_input_points
		self.sample_dist = sample_dist
		self.surface_uniform_ratio = surface_uniform_ratio
		self.device = device
		self.decoder_layers = decoder_layers
		self.predict_blending = predict_blending
		self.predict_roundness = predict_roundness
		self.no_batch_norm = no_batch_norm

		self.point_encoder = PointNetfeat(global_feat=True, no_batch_norm=no_batch_norm)
		self.regressor_decoder_list = nn.ModuleList()

		# Initialize a separate decoder for each primitive
		for i in range(self.num_prims):
			regressor_decoder = PrimitiveRegressor(
				POINTNET_FEAT_OUTPUT_SIZE,
				self.num_shapes,
				self.num_operations,
				layer_sizes=self.decoder_layers,
				predict_blending=self.predict_blending,
				predict_roundness=self.predict_roundness,
				no_batch_norm=self.no_batch_norm
			)

			self.regressor_decoder_list.append(regressor_decoder)

		self.to(self.device)


	def forward(self, target_input_samples, csg_model=None):
		batch_size = target_input_samples.size(dim=0)
		num_points = target_input_samples.size(dim=1)
		first_prim = csg_model is None

		if first_prim:
			# Insert a tensor of shape BxNx1 to represent the SDF values for the null initial reconstruction
			csg_model = CSGModel(batch_size, device=self.device)
			init_recon_sdf = torch.ones((batch_size, num_points, 1), device=self.device)
		else:
			# Sample the CSG model
			target_points = target_input_samples[:,:,:3]
			init_recon_sdf = csg_model.sample_csg(target_points).unsqueeze(-1)

		combined_samples = torch.cat((target_input_samples, init_recon_sdf), -1)

		# Change input shape from BxNx4 to Bx4xN for PointNet encoder
		# Where B = Batch Size and N = Number of Points
		combined_samples = combined_samples.permute(0, 2, 1)

		# Encode target point cloud features
		features, _, _ = self.point_encoder(combined_samples)

		# Decode primitive predictions
		for decoder in self.regressor_decoder_list:
			csg_model.add_command(*decoder(features, first_prim))
			first_prim = False

		return csg_model


	def forward_cascade(self, target_input_samples, num_cascades):
		csg_model = self.forward(target_input_samples)

		for i in range(num_cascades):
			csg_model = self.forward(target_input_samples, csg_model)

		return csg_model


	# Set which operations to scale and by how much
	def set_operation_weight(self, scale_op, replace_op, operation_scale):
		for regressor_decoder in self.regressor_decoder_list:
			regressor_decoder.set_operation_weight(scale_op, replace_op, operation_scale)


# Test network
def test():
	# Input Dimension: BxFxP
	# B = Batch Size
	# F = Feature Size
	# P = Point Size
	target_points = torch.rand(32, 1024, 4)
	initial_points = torch.rand(32, 1024, 4)

	csg_crn = CSG_CRN(3, 2)
	outputs = csg_crn(target_points, initial_points)
	print('Network Output Size:', len(outputs))