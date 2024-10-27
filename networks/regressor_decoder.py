import torch
import torch.nn as nn


# Dimensions of the layers in each regressor
REGRESSOR_LAYER_SIZES = [256]
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
	def __init__(self, layer_sizes, activation_func=None):
		super(RegressorNetwork, self).__init__()
		self.layer_sizes = layer_sizes
		self.activation_func = activation_func
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

		return X


# Predict probability distribution for output shape primitive
class ShapeRegressor(RegressorNetwork):
	def __init__(self, layer_sizes, num_shapes):
		layer_sizes.append(num_shapes)
		activation = nn.Softmax(dim=-1)
		super(ShapeRegressor, self).__init__(layer_sizes, activation)


# Predict probability distribution for boolean operation to apply
class OperationRegressor(RegressorNetwork):
	def __init__(self, layer_sizes, num_operations):
		layer_sizes.append(num_operations)
		activation = nn.Softmax(dim=-1)
		super(OperationRegressor, self).__init__(layer_sizes, activation)


# Predict 3D coordinate
class TranslationRegressor(RegressorNetwork):
	def __init__(self, layer_sizes, translation_scale):
		self.translation_scale = translation_scale
		layer_sizes.append(3)
		activation = nn.Tanh()
		super(TranslationRegressor, self).__init__(layer_sizes, activation)

	# Restrict predicted coordinates to fit the output unit cube
	def forward(self, X):
		translation = super(TranslationRegressor, self).forward(X)
		return translation * self.translation_scale


# Predict quaternion rotation
class RotationRegressor(RegressorNetwork):
	def __init__(self, layer_sizes):
		layer_sizes.append(4)
		super(RotationRegressor, self).__init__(layer_sizes)

	# Normalize quaternion
	def forward(self, X):
		quaternion = super(RotationRegressor, self).forward(X)
		return torch.nn.functional.normalize(X, dim=-1)


# Predict shape scale in x, y, and z axes
class ScaleRegressor(RegressorNetwork):
	def __init__(self, layer_sizes, min_scale, max_scale):
		self.min_scale = min_scale
		self.max_scale = max_scale
		layer_sizes.append(3)
		activation = torch.sigmoid
		super(ScaleRegressor, self).__init__(layer_sizes, activation)

	# Restrict the predicted scale to expected range
	def forward(self, X):
		scale = super(ScaleRegressor, self).forward(X)
		return (scale * (self.max_scale - self.min_scale)) + self.min_scale


# Predict amount to blend output primitive with current reconstruction
class BlendingRegressor(RegressorNetwork):
	def __init__(self, layer_sizes, min_blending, max_blending):
		self.min_blending = min_blending
		self.max_blending = max_blending
		layer_sizes.append(1)
		activation = torch.sigmoid
		super(BlendingRegressor, self).__init__(layer_sizes, activation)
	
	# Restrict the blending factor to expected range
	def forward(self, X):
		blending = super(BlendingRegressor, self).forward(X)
		return (blending * (self.max_blending - self.min_blending)) + self.min_blending


# Predict amount to round shape into sphere
class RoundnessRegressor(RegressorNetwork):
	def __init__(self, layer_sizes):
		layer_sizes.append(1)
		activation = torch.sigmoid
		super(RoundnessRegressor, self).__init__(layer_sizes, activation)


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

		self.shape = ShapeRegressor(REGRESSOR_LAYER_SIZES, num_shapes)
		self.operation = OperationRegressor(REGRESSOR_LAYER_SIZES, num_operations)
		self.translation = TranslationRegressor(REGRESSOR_LAYER_SIZES, translation_scale)
		self.rotation = RotationRegressor(REGRESSOR_LAYER_SIZES)
		self.scale = ScaleRegressor(REGRESSOR_LAYER_SIZES, min_scale, max_scale)
		self.blending = BlendingRegressor(REGRESSOR_LAYER_SIZES, min_blending, max_blending) if (predict_blending) else (None)
		self.roundness = RoundnessRegressor(REGRESSOR_LAYER_SIZES) if (predict_roundness) else (None)
	
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

	network = ShapeRegressor(REGRESSOR_LAYER_SIZES, 3)
	outputs = network(inputs)
	print('Shape Output:')
	print(outputs, '\n')
	
	network = OperationRegressor(REGRESSOR_LAYER_SIZES, 2)
	outputs = network(inputs)
	print('Boolean Regressor Output:')
	print(outputs, '\n')
	
	network = TranslationRegressor(REGRESSOR_LAYER_SIZES, 0.6)
	outputs = network(inputs)
	print('Translation Output:')
	print(outputs, '\n')
	
	network = RotationRegressor(REGRESSOR_LAYER_SIZES)
	outputs = network(inputs)
	print('Rotation Output:')
	print(outputs, '\n')

	network = ScaleRegressor(REGRESSOR_LAYER_SIZES, 0.005, 0.5)
	outputs = network(inputs)
	print('Scale Output:')
	print(outputs, '\n')

	network = BlendingRegressor(REGRESSOR_LAYER_SIZES, 0, 1)
	outputs = network(inputs)
	print('Blending Output:')
	print(outputs, '\n')

	network = RoundnessRegressor(REGRESSOR_LAYER_SIZES)
	outputs = network(inputs)
	print('Roundness Output:')
	print(outputs, '\n')

	network = PrimitiveRegressor(REGRESSOR_LAYER_SIZES, 3, 2)
	outputs = network(inputs)
	print('Combined Output Size:', len(outputs))