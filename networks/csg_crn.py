import torch
import torch.nn as nn
from networks.pointnet import PointNetfeat, POINTNET_FEAT_OUTPUT_SIZE
from networks.siamese_encoder import SiameseEncoder, SIAMEZE_ENCODER_OUTPUT_SIZE
from networks.regressor_decoder import PrimitiveRegressor


class CSG_CRN(nn.Module):
	def __init__(self, num_prims, num_shapes, num_operations, decoder_layers=[],
		predict_blending=True, predict_roundness=True, no_batch_norm=False):

		super(CSG_CRN, self).__init__()
		self.num_prims = num_prims
		self.num_shapes = num_shapes
		self.num_operations = num_operations
		self.predict_blending = predict_blending
		self.predict_roundness = predict_roundness

		self.point_encoder = PointNetfeat(global_feat=True, no_batch_norm=no_batch_norm)
		self.siamese_encoder = SiameseEncoder(self.point_encoder, POINTNET_FEAT_OUTPUT_SIZE, no_batch_norm)
		self.regressor_decoder_list = nn.ModuleList()

		# Initialize a separate decoder for each primitive
		for i in range(self.num_prims):
			regressor_decoder = PrimitiveRegressor(SIAMEZE_ENCODER_OUTPUT_SIZE, self.num_shapes, self.num_operations,
				layer_sizes=decoder_layers, predict_blending=self.predict_blending, predict_roundness=self.predict_roundness, no_batch_norm=no_batch_norm)
			self.regressor_decoder_list.append(regressor_decoder)


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