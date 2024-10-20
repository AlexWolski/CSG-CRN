import torch
import torch.nn as nn


# Defines the layers number and count for the siamese encoder
LAYER_SIZES = [1024, 512, 512, 256]


# Encode two inputs with a weight-sharing siamese encoder and learn combined feature vector
class SiameseEncoder(nn.Module):
	def __init__(self, encoder, encoder_feature_size, no_batch_norm=False):
		super(SiameseEncoder, self).__init__()
		self.encoder = encoder
		self.encoder_feature_size = encoder_feature_size

		self.relu = nn.ReLU()
		self.fc_list = nn.ModuleList()
		self.bn_list = nn.ModuleList()
		self.num_layers = len(LAYER_SIZES)

		# Initialize layers
		# Contrastive encoder inspired by PCRNet
		# https://arxiv.org/abs/1908.07906
		for i in range(self.num_layers):
			initial_layer_size = encoder_feature_size*2
			prev_layer_size = LAYER_SIZES[i-1] if i > 0 else initial_layer_size
			curr_layer_size = LAYER_SIZES[i]
			batch_norm_layer = nn.Identity() if no_batch_norm else nn.BatchNorm1d(curr_layer_size)

			self.fc_list.append(nn.Linear(prev_layer_size, curr_layer_size))
			self.bn_list.append(batch_norm_layer)


	def forward(self, target_input, initial_recon_input=None):
		# Weights are shared in encoders
		target_features, _, _ = self.encoder(target_input)

		# If no initial_recon_input is provided, set the output features to a zero tensor
		if initial_recon_input is not None:
			initial_recon_features, _, _ = self.encoder(initial_recon_input)
		else:
			initial_recon_features = torch.zeros(target_features.size()).to(target_input.device)

		features = torch.cat([target_features, initial_recon_features], dim=1)

		for i in range(self.num_layers):
			fc = self.fc_list[i]
			bn = self.bn_list[i]
			features = bn(self.relu(fc(features)))

		return features


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