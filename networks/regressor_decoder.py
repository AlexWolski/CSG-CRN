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
	def __init__(self, layer_sizes, activation_func=None, normalization_func=None):
		super(RegressorNetwork, self).__init__()
		self.layer_sizes = layer_sizes
		self.activation_func = activation_func
		self.normalization_func = normalization_func
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.init_layers();

	def init_layers(self):
		self.fc_list = nn.ModuleList()

		for i in range(len(self.layer_sizes) - 1):
			self.fc_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))

	def forward(self, X):
		for i in range(len(self.fc_list)):
			fc_layer = self.fc_list[i]
			X = fc_layer(X)

			if i < len(self.fc_list) - 1:
				X = self.LeReLU(X)
			elif self.activation_func != None:
				X = self.activation_func(X)

		if self.normalization_func != None:
			X = self.normalization_func(X)

		return X


def normalizeTranslation(translation_scale):
	return lambda X: X * translation_scale

def normalizeRotation():
	return lambda X: torch.nn.functional.normalize(X, dim=-1)

def normalizeScale(min_scale, max_scale):
	return lambda X: (X * (max_scale - min_scale)) + min_scale

def normalizeBlending(min_blending, max_blending):
	return lambda X: (X * (max_blending - min_blending)) + min_blending


# Pridict all primitive parameters 
class PrimitiveRegressor(nn.Module):
	def __init__(self,
		num_shapes, num_operations,
		translation_scale=DEFAULT_TRANSLATION_SCALE,
		min_scale=DEFAULT_MIN_SCALE,
		max_scale=DEFAULT_MAX_SCALE,
		predict_blending=True,
		min_blending=DEFAULT_MIN_BLENDING,
		max_blending=DEFAULT_MAX_BLENDING,
		predict_roundness=True):

		super(PrimitiveRegressor, self).__init__()

		self.shape = RegressorNetwork([256, num_shapes], nn.Softmax(dim=-1))
		self.operation = RegressorNetwork([256, num_operations], nn.Softmax(dim=-1))
		self.translation = RegressorNetwork([256, 3], nn.Tanh(), normalizeTranslation(translation_scale))
		self.rotation = RegressorNetwork([256, 4], None, normalizeRotation())
		self.scale = RegressorNetwork([256, 3], torch.sigmoid, normalizeScale(min_scale, max_scale))
		self.blending = RegressorNetwork([256, 1], torch.sigmoid, normalizeBlending(min_blending, max_blending)) if (predict_blending) else (None)
		self.roundness = RegressorNetwork([256, 1], torch.sigmoid) if (predict_roundness) else (None)
	
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