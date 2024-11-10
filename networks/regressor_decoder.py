import torch
import torch.nn as nn


# Tune Leaky ReLU slope for predicting negative values
LEAKY_RELU_NEGATIVE_SLOPE = 0.2

# Default settings for regressor decoder
DEFAULT_TRANSLATION_SCALE = 1
DEFAULT_MIN_SCALE = 0.005
DEFAULT_MAX_SCALE = 1
DEFAULT_MIN_BLENDING = 0.001
DEFAULT_MAX_BLENDING = 1


# Parent regressor network class to generalize network building
class RegressorNetwork(nn.Module):
	def __init__(self, layer_sizes, activation_func=None, normalization_func=None, no_batch_norm=False):
		super(RegressorNetwork, self).__init__()
		self.layer_sizes = layer_sizes
		self.activation_func = activation_func
		self.normalization_func = normalization_func
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.init_layers(no_batch_norm);

	def init_layers(self, no_batch_norm=False):
		self.fc_list = nn.ModuleList()
		self.bn_list = nn.ModuleList() if not no_batch_norm else None

		for i in range(len(self.layer_sizes) - 1):
			prev_layer_size = self.layer_sizes[i]
			curr_layer_size = self.layer_sizes[i+1]
			self.fc_list.append(nn.Linear(prev_layer_size, curr_layer_size))

			if self.bn_list != None and i+1 < len(self.layer_sizes) - 1:
				self.bn_list.append(nn.BatchNorm1d(curr_layer_size))

	def forward(self, X):
		for i in range(len(self.fc_list)):
			fc_layer = self.fc_list[i]
			X = fc_layer(X)

			if i < len(self.fc_list) - 1:
				X = self.LeReLU(X)

				if self.bn_list != None:
					bn_layer = self.bn_list[i]
					X = bn_layer(X)

			elif self.activation_func != None:
				X = self.activation_func(X)

		if self.normalization_func != None:
			X = self.normalization_func(X)

		return X


# Pridict all primitive parameters 
class PrimitiveRegressor(nn.Module):
	def _normalizeTranslation(self, translation_scale):
		return lambda X: X * translation_scale

	def _normalizeRotation(self):
		return lambda X: torch.nn.functional.normalize(X, dim=-1)

	def _normalizeScale(self, min_scale, max_scale):
		return lambda X: (X * (max_scale - min_scale)) + min_scale

	def _normalizeBlending(self, min_blending, max_blending):
		return lambda X: (X * (max_blending - min_blending)) + min_blending


	def __init__(self,
		input_feature_size,
		num_shapes, num_operations,
		translation_scale=DEFAULT_TRANSLATION_SCALE,
		min_scale=DEFAULT_MIN_SCALE,
		max_scale=DEFAULT_MAX_SCALE,
		predict_blending=True,
		min_blending=DEFAULT_MIN_BLENDING,
		max_blending=DEFAULT_MAX_BLENDING,
		predict_roundness=True,
		no_batch_norm=False):

		super(PrimitiveRegressor, self).__init__()

		self.shape = RegressorNetwork([input_feature_size, num_shapes], nn.Softmax(dim=-1), no_batch_norm=no_batch_norm)
		self.operation = RegressorNetwork([input_feature_size, num_operations], nn.Softmax(dim=-1), no_batch_norm=no_batch_norm)
		self.translation = RegressorNetwork([input_feature_size, 3], nn.Tanh(), self._normalizeTranslation(translation_scale), no_batch_norm=no_batch_norm)
		self.rotation = RegressorNetwork([input_feature_size, 4], None, self._normalizeRotation(), no_batch_norm=no_batch_norm)
		self.scale = RegressorNetwork([input_feature_size, 3], torch.sigmoid, self._normalizeScale(min_scale, max_scale), no_batch_norm=no_batch_norm)
		self.blending = RegressorNetwork([input_feature_size, 1], torch.sigmoid, self._normalizeBlending(min_blending, max_blending), no_batch_norm=no_batch_norm) if (predict_blending) else (None)
		self.roundness = RegressorNetwork([input_feature_size, 1], torch.sigmoid, no_batch_norm=no_batch_norm) if (predict_roundness) else (None)

	
	def forward(self, X, has_initial_recon):
		shape = self.shape.forward(X)
		operation = self.operation.forward(X)
		translation = self.translation.forward(X)
		rotation = self.rotation.forward(X)
		scale = self.scale.forward(X)
		blending = self.blending.forward(X) if (self.blending is not None and has_initial_recon) else (None)
		roundness = self.roundness.forward(X) if (self.roundness is not None) else (None)

		return(
			shape,
			operation,
			translation,
			rotation,
			scale,
			blending,
			roundness
		)


# Test network
def test():
	batch_size = 2
	feature_size = 256
	inputs = torch.autograd.Variable(torch.rand(batch_size, feature_size))

	network = PrimitiveRegressor(REGRESSOR_LAYER_SIZES, 3, 2)
	outputs = network(inputs)
	print('Combined Output Size:', len(outputs))