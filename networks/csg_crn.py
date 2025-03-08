import torch
import torch.nn as nn
from networks.pointnet import PointNetfeat, POINTNET_FEAT_OUTPUT_SIZE
from networks.siamese_encoder import SiameseEncoder, SIAMEZE_ENCODER_OUTPUT_SIZE
from networks.regressor_decoder import PrimitiveRegressor
from utilities.csg_model import CSGModel
from utilities.sampler_utils import sample_sdf_from_csg_combined


class CSG_CRN(nn.Module):
	def __init__(
			self, num_prims, num_shapes, num_operations, num_cascades, sample_dist, surface_uniform_ratio, device,
			decoder_layers=[], predict_blending=True, predict_roundness=True, no_batch_norm=False):
		super(CSG_CRN, self).__init__()

		self.num_prims = num_prims
		self.num_shapes = num_shapes
		self.num_operations = num_operations
		self.num_cascades = num_cascades
		self.sample_dist = sample_dist
		self.surface_uniform_ratio = surface_uniform_ratio
		self.device = device
		self.decoder_layers = decoder_layers
		self.predict_blending = predict_blending
		self.predict_roundness = predict_roundness
		self.no_batch_norm = no_batch_norm

		self.point_encoder = PointNetfeat(global_feat=True, no_batch_norm=no_batch_norm)
		self.siamese_encoder = SiameseEncoder(self.point_encoder, POINTNET_FEAT_OUTPUT_SIZE, no_batch_norm)
		self.regressor_decoder_list = nn.ModuleList()

		# Initialize a separate decoder for each primitive
		for i in range(self.num_prims):
			regressor_decoder = PrimitiveRegressor(
				SIAMEZE_ENCODER_OUTPUT_SIZE,
				self.num_shapes,
				self.num_operations,
				layer_sizes=self.decoder_layers,
				predict_blending=self.predict_blending,
				predict_roundness=self.predict_roundness,
				no_batch_norm=self.no_batch_norm
			)

			self.regressor_decoder_list.append(regressor_decoder)

		self.to(self.device)


	def forward(self, target_input, initial_recon_input=None):
		# Change input shape from BxNx4 to Bx4xN for PointNet encoder
		# Where B = Batch Size and N = Number of Points
		target_input = target_input.permute(0, 2, 1)

		if initial_recon_input is not None:
			initial_recon_input = initial_recon_input.permute(0, 2, 1)

		features = self.siamese_encoder(target_input, initial_recon_input)

		output_list = []
		first_prim = True

		for i, decoder in enumerate(self.regressor_decoder_list):
			output_list.append(decoder(features, first_prim))
			first_prim = False

		return output_list


	def forward_cascade(self, target_input_samples):
		# Initialize SDF CSG model
		(batch_size, num_input_points, _) = target_input_samples.size()
		csg_model = CSGModel(self.device)
		recon_input_samples = None

		for i in range(self.num_cascades):
			output_list = self(target_input_samples, recon_input_samples)

			# Add primitives to the CSG model
			for output in output_list:
				csg_model.add_command(*output)

			# Sample CSG for next refinement loop
			if i < self.num_cascades - 1:
				(recon_input_points, recon_input_distances) = sample_sdf_from_csg_combined(csg_model, num_input_points, self.sample_dist, self.surface_uniform_ratio)
				recon_input_samples = torch.cat((recon_input_points, recon_input_distances.unsqueeze(2)), dim=-1)

		return csg_model


	# Set which operations to scale and by how much
	def set_operation_scale(self, scale_op, replace_op, operation_scale):
		for regressor_decoder in self.regressor_decoder_list:
			regressor_decoder.set_operation_scale(scale_op, replace_op, operation_scale)


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