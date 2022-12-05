import torch
import torch.nn as nn
import pointnet
import siamese_encoder
import regressor_decoder


class CSG_CRN(nn.Module):
	def __init__(self, num_primitives, num_operations):
		super(CSG_CRN, self).__init__()
		self.num_primitives = num_primitives
		self.num_operations = num_operations

		self.point_encoder = pointnet.PointNetfeat(global_feat=True)
		self.siamese_encoder = siamese_encoder.SiameseEncoder(self.point_encoder, 1024)
		self.regressor_decoder = regressor_decoder.PrimitiveRegressor(num_primitives, num_operations)


	def forward(self, target_input, initial_recon_input):
		features = self.siamese_encoder(target_input, initial_recon_input)
		outputs = self.regressor_decoder(features)

		return outputs


# Test network
if __name__ == '__main__':
	import pointnet

	# Input Dimension: BxFxP
	# B = Batch Size
	# F = Feature Size
	# P = Point Size
	target_points = torch.autograd.Variable(torch.rand(32, 4, 1024))
	initial_points = torch.autograd.Variable(torch.rand(32, 4, 1024))

	csg_crn = CSG_CRN(3, 2)
	outputs = csg_crn(target_points, initial_points)
	print('Network Output Size:', len(outputs))