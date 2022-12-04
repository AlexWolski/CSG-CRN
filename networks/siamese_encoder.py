import torch
import torch.nn as nn


# Tune Leaky ReLU slope for predicting negative values
LEAKY_RELU_NEGATIVE_SLOPE = 0.2


# Encode two inputs with a weight-sharing siamese encoder and learn combined feature vector
class SiameseEncoder(nn.Module):
	def __init__(self, encoder, encoder_feature_size):
		super(SiameseEncoder, self).__init__()
		self.encoder = encoder
		self.encoder_feature_size = encoder_feature_size

		# Contrastive encoder inspired by PCRNet
		# https://arxiv.org/abs/1908.07906
		self.fc1 = nn.Linear(encoder_feature_size*2, 1024)
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 512)
		self.fc4 = nn.Linear(512, 256)
		self.bn1 = nn.BatchNorm1d(1024)
		self.bn2 = nn.BatchNorm1d(512)
		self.bn3 = nn.BatchNorm1d(512)
		self.bn4 = nn.BatchNorm1d(256)
		self.nonLinear = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)


	def forward(self, target_input, initial_recon_input):
		# Weights are shared in encoders
		target_features, _, _ = self.encoder(target_input)
		initial_recon_features, _, _ = self.encoder(initial_recon_input)

		combined_features = torch.cat([target_features, initial_recon_features], dim=1)

		X = self.bn1(self.nonLinear(self.fc1(combined_features)))
		X = self.bn2(self.nonLinear(self.fc2(X)))
		X = self.bn3(self.nonLinear(self.fc3(X)))
		final_features = self.bn4(self.nonLinear(self.fc4(X)))

		return final_features


# Test network
if __name__ == '__main__':
	import pointnet

	# Input Dimension: BxFxP
	# B = Batch Size
	# F = Feature Size
	# P = Point Size
	target_points = torch.autograd.Variable(torch.rand(32, 4, 1024))
	initial_points = torch.autograd.Variable(torch.rand(32, 4, 1024))

	pointnet_encoder = pointnet.PointNetfeat(global_feat=True)
	SiameseEncoder = SiameseEncoder(pointnet_encoder, 1024)
	outputs = SiameseEncoder(target_points, initial_points)
	print('Encoder Output Size:', outputs.size())