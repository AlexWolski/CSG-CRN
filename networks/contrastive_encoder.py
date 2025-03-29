import torch
import torch.nn as nn
from networks.pointnet import POINTNET_FEAT_OUTPUT_SIZE


# Defines the layers number and count for the contrastive encoder
INPUT_SIZE = POINTNET_FEAT_OUTPUT_SIZE * 2
INTERMEDIATE_SIZE = int(POINTNET_FEAT_OUTPUT_SIZE * 1.5)
LAYER_SIZES = [INPUT_SIZE, INTERMEDIATE_SIZE, INTERMEDIATE_SIZE, INTERMEDIATE_SIZE, POINTNET_FEAT_OUTPUT_SIZE]


# Encode two inputs with a weight-sharing contrastive encoder and learn combined feature vector
class ContrastiveEncoder(nn.Module):
	def __init__(self, encoder_feature_size, no_batch_norm=False):
		super(ContrastiveEncoder, self).__init__()
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


	def forward(self, target_features, recon_features):
		features = torch.cat([target_features, recon_features], dim=1)

		# Apply fully connected, relu, and batch normalization layers
		for i in range(self.num_layers):
			fc_layer = self.fc_list[i]
			features = self.relu(fc_layer(features))

			# Apply batch normalization
			if self.bn_list != None:
				bn_layer = self.bn_list[i]
				features = bn_layer(features)

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
	contrastive_encoder = ContrastiveEncoder(pointnet_encoder, 1024)
	outputs = contrastive_encoder(target_points, initial_points)
	print('Encoder Output Size:', outputs.size())
