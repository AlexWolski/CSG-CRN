import torch
import torch.nn as nn


# Encode two inputs with a weight-sharing siamese encoder and learn combined feature vector
class SiameseEncoder(nn.Module):
	def __init__(self, encoder, encoder_feature_size, no_batch_norm=False):
		super(SiameseEncoder, self).__init__()
		self.encoder = encoder
		self.encoder_feature_size = encoder_feature_size

		# Contrastive encoder inspired by PCRNet
		# https://arxiv.org/abs/1908.07906
		self.fc1 = nn.Linear(encoder_feature_size*2, 1024)
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 512)
		self.fc4 = nn.Linear(512, 256)

		if no_batch_norm:
			self.bn1 = nn.Identity()
			self.bn2 = nn.Identity()
			self.bn3 = nn.Identity()
			self.bn4 = nn.Identity()
		else:
			self.bn1 = nn.BatchNorm1d(1024)
			self.bn2 = nn.BatchNorm1d(512)
			self.bn3 = nn.BatchNorm1d(512)
			self.bn4 = nn.BatchNorm1d(256)

		self.relu = nn.ReLU()


	def forward(self, target_input, initial_recon_input=None):
		# Weights are shared in encoders
		target_features, _, _ = self.encoder(target_input)

		# If no initial_recon_input is provided, set the output features to a zero tensor
		if initial_recon_input is not None:
			initial_recon_features, _, _ = self.encoder(initial_recon_input)
		else:
			initial_recon_features = torch.zeros(target_features.size()).to(target_input.device)

		combined_features = torch.cat([target_features, initial_recon_features], dim=1)

		X = self.bn1(self.relu(self.fc1(combined_features)))
		X = self.bn2(self.relu(self.fc2(X)))
		X = self.bn3(self.relu(self.fc3(X)))
		contrastive_features = self.bn4(self.relu(self.fc4(X)))

		return contrastive_features


# Test network
def test():
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