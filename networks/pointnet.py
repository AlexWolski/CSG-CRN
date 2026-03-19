# Modified from pointnet.pytorch github repository
# https://github.com/fxia22/pointnet.pytorch
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


POINTNET_CONV_LAYER_SIZES = [64, 128, 1024]
POINTNET_FEAT_OUTPUT_SIZE = POINTNET_CONV_LAYER_SIZES[-1]
TRANS_CONV_LAYER_SIZES = [64, 128, 1024]
TRANS_FC_LAYER_SIZES = [512, 256]


class STNkd(nn.Module):
	def __init__(self, k, no_batch_norm=False):
		super(STNkd, self).__init__()
		self.conv_layer_sizes = [k] + TRANS_CONV_LAYER_SIZES
		self.fc_layer_sizes = ([self.conv_layer_sizes[-1]] if len(self.conv_layer_sizes) else []) + TRANS_FC_LAYER_SIZES + [k*k]
		self.k = k
		self.no_batch_norm = no_batch_norm
		self._init_layers()


	def _init_layers(self):
		self.conv_list = nn.ModuleList()
		self.fc_list = nn.ModuleList()
		self.bn_list = nn.ModuleList() if not self.no_batch_norm else None
		self.relu = nn.ReLU()

		# Initialize convolutional and batch normalization layers
		for i in range(len(self.conv_layer_sizes) - 1):
			prev_layer_size = self.conv_layer_sizes[i]
			curr_layer_size = self.conv_layer_sizes[i+1]
			self.conv_list.append(torch.nn.Conv1d(prev_layer_size, curr_layer_size, 1))

			if not self.no_batch_norm:
				self.bn_list.append(nn.BatchNorm1d(curr_layer_size))

		# Initialize fully connected and batch normalization layers.
		# Do not apply batch normalization to the last layer.
		for i in range(len(self.fc_layer_sizes) - 1):
			prev_layer_size = self.fc_layer_sizes[i]
			curr_layer_size = self.fc_layer_sizes[i+1]
			self.fc_list.append(nn.Linear(prev_layer_size, curr_layer_size))

			if not self.no_batch_norm:
				is_last_layer = i+1 == len(self.fc_layer_sizes) - 1
				bn_layer = nn.BatchNorm1d(curr_layer_size) if not is_last_layer else nn.Identity()
				self.bn_list.append(bn_layer)


	def forward(self, X):
		batchsize = X.size()[0]
		bn_index = 0

		# Forward through convolutional layers
		for conv_layer in self.conv_list:
			bn_layer = self.bn_list[bn_index] if not self.no_batch_norm else nn.Identity()
			bn_index += 1
			X = self.relu(bn_layer(conv_layer(X)))

		# Double check convolutional layers exist, apply max pooling, and reshape input for fully connected layers
		if len(self.conv_list):
			X = torch.max(X, 2, keepdim=True)[0]
			X = X.view(-1, self.conv_list[-1].out_channels)
 
		# Forward through fully connected layers
		for fc_layer in self.fc_list[:-1]:
			bn_layer = self.bn_list[bn_index] if not self.no_batch_norm else nn.Identity()
			bn_index += 1
			X = self.relu(bn_layer(fc_layer(X)))

		# Last fully connected layer does not have batch normalization or relu activation
		if len(self.fc_list):
			X = self.fc_list[-1](X)

		iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)

		if X.is_cuda:
			iden = iden.cuda()

		X = X + iden
		X = X.view(-1, self.k, self.k)
		return X


class PointNetfeat(nn.Module):
	def __init__(self, k=6, global_feat=True, input_transform=False, feature_transform=False, no_batch_norm=False):
		super(PointNetfeat, self).__init__()
		self.conv_layer_sizes = [k] + POINTNET_CONV_LAYER_SIZES
		self.global_feat = global_feat
		self.input_transform = input_transform
		self.feature_transform = feature_transform
		self.no_batch_norm = no_batch_norm
		self._init_layers()

		if self.input_transform:
			self.stn = STNkd(k=k, no_batch_norm=no_batch_norm)

		if self.feature_transform:
			self.fstn = STNkd(k=64, no_batch_norm=no_batch_norm)


	def _init_layers(self):
		self.conv_list = nn.ModuleList()
		self.bn_list = nn.ModuleList() if not self.no_batch_norm else None
		self.relu = nn.ReLU()

		# Initialize convolutional and batch normalization layers
		for i in range(len(self.conv_layer_sizes) - 1):
			prev_layer_size = self.conv_layer_sizes[i]
			curr_layer_size = self.conv_layer_sizes[i+1]
			self.conv_list.append(torch.nn.Conv1d(prev_layer_size, curr_layer_size, 1))

			if not self.no_batch_norm:
				self.bn_list.append(nn.BatchNorm1d(curr_layer_size))


	def forward(self, X):
		n_pts = X.size()[2]

		if self.input_transform:
			trans_input = self.stn(X)
			X = X.transpose(2, 1)
			X = torch.bmm(X, trans_input)
			X = X.transpose(2, 1)
		else:
			trans_input = None

		# First convolutional layer
		if len(self.conv_list):
			bn_layer = self.bn_list[0] if not self.no_batch_norm else nn.Identity()
			X = F.relu(bn_layer(self.conv_list[0](X)))

		if self.feature_transform:
			trans_feat = self.fstn(X)
			X = X.transpose(2,1)
			X = torch.bmm(X, trans_feat)
			X = X.transpose(2,1)
		else:
			trans_feat = None

		pointfeat = X
		bn_index = 1

		# Middle convolutional layers
		for conv_layer in self.conv_list[1:-1]:
			bn_layer = self.bn_list[bn_index] if not self.no_batch_norm else nn.Identity()
			bn_index += 1
			X = self.relu(bn_layer(conv_layer(X)))

		# Last convolutional layer
		if len(self.conv_list):
			X = self.bn_list[-1](self.conv_list[-1](X))

		X = torch.max(X, 2, keepdim=True)[0]
		X = X.view(-1, 1024)

		if self.global_feat:
			return X, trans_input, trans_feat
		else:
			X = X.view(-1, 1024, 1).repeat(1, 1, n_pts)
			return torch.cat([X, pointfeat], 1), trans_input, trans_feat


def feature_transform_regularizer(trans):
	d = trans.size()[1]
	I = torch.eye(d)[None, :, :]
	if trans.is_cuda:
		I = I.cuda()
	loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
	return loss


if __name__ == '__main__':
	sim_data = Variable(torch.rand(32,5,2500))
	trans = STNkd(5)
	out = trans(sim_data)
	print('stn', out.size())
	print('loss', feature_transform_regularizer(out))

	pointfeat = PointNetfeat(global_feat=True)
	out, _, _ = pointfeat(sim_data)
	print('global feat', out.size())

	pointfeat = PointNetfeat(global_feat=False)
	out, _, _ = pointfeat(sim_data)
	print('point feat', out.size())
