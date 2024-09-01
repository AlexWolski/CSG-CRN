import torch
import torch.nn as nn
from networks.pointnet import PointNetfeat
from networks.siamese_encoder import SiameseEncoder
from networks.regressor_decoder import PrimitiveRegressor


class CSG_CRN(nn.Module):
	def __init__(self, num_shapes, num_operations,
		predict_blending=True, predict_roundness=True, no_batch_norm=False):

		super(CSG_CRN, self).__init__()
		self.num_shapes = num_shapes
		self.num_operations = num_operations
		self.predict_blending = predict_blending
		self.predict_roundness = predict_roundness

		self.point_encoder = PointNetfeat(global_feat=True, no_batch_norm=no_batch_norm)
		self.siamese_encoder = SiameseEncoder(self.point_encoder, 1024, no_batch_norm)
		self.regressor_decoder = PrimitiveRegressor(self.num_shapes, self.num_operations,
			predict_blending=self.predict_blending, predict_roundness=self.predict_roundness)


	def forward(self, target_input, initial_recon_input=None):
		# Change input shape from BxNx4 to Bx4xN for PointNet encoder
		# Where B = Batch Size and N = Number of Points
		target_input = target_input.permute(0, 2, 1)

		if initial_recon_input is not None:
			initial_recon_input = initial_recon_input.permute(0, 2, 1)

		features = self.siamese_encoder(target_input, initial_recon_input)
		outputs = self.regressor_decoder(features, initial_recon_input is not None)

		return outputs


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