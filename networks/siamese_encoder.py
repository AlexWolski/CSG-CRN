import torch
import torch.nn as nn


# Defines the layers number and count for the siamese encoder
LAYER_SIZES = [1024, 512, 512, 512, 256]
SIAMEZE_ENCODER_OUTPUT_SIZE = LAYER_SIZES[-1]


# Encode two inputs with a weight-sharing siamese encoder and learn combined feature vector
class SiameseEncoder(nn.Module):
	def __init__(self, encoder, encoder_feature_size, no_batch_norm=False):
		super(SiameseEncoder, self).__init__()
		self.encoder = encoder

		self.relu = nn.ReLU()
		self.fc_list = nn.ModuleList()
		self.bn_list = nn.ModuleList() if not no_batch_norm else None
		self.num_layers = len(LAYER_SIZES)

		# Initialize layers
		# Contrastive encoder inspired by PCRNet
		# https://arxiv.org/abs/1908.07906
		for i in range(self.num_layers):
			initial_layer_size = encoder_feature_size*2
			prev_layer_size = LAYER_SIZES[i-1] if i > 0 else initial_layer_size
			curr_layer_size = LAYER_SIZES[i]
			self.fc_list.append(nn.Linear(prev_layer_size, curr_layer_size))

			if self.bn_list != None:
				self.bn_list.append(nn.BatchNorm1d(curr_layer_size))


	def forward(self, target_features, initial_recon_features):
		features = torch.cat([target_features, initial_recon_features], dim=1)

		# Apply fully connected, relu, and batch normalization layers
		for i in range(self.num_layers):
			fc_layer = self.fc_list[i]
			features = self.relu(fc_layer(features))

			# Apply batch normalization
			if self.bn_list != None:
				bn_layer = self.bn_list[i]
				features = bn_layer(features)

		return features


	# Encode the features of a target point cloud
	def forward_initial(self, target_input):
		# Weights are shared in encoders
		target_features, _, _ = self.encoder(target_input)
		initial_recon_features = torch.zeros(target_features.size(), device=target_input.device)
		features = self.forward(target_features, initial_recon_features)
		return features, target_features


	# Encode the contrastive features between target features and an initial reconstruction point cloud
	def forward_refine(self, target_features, initial_recon_input):
		initial_recon_features, _, _ = self.encoder(initial_recon_input)
		return self.forward(target_features.detach(), initial_recon_features)


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