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


# Predict probability distribution for output shape primitive
class ShapeRegressor(nn.Module):
	def __init__(self, num_shapes):
		super(ShapeRegressor, self).__init__()
		self.num_shapes = num_shapes
		self.fc_list = nn.ModuleList()
		self.num_layers = len(REGRESSOR_LAYER_SIZES)

		for i in range(self.num_layers - 1):
			self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[i], REGRESSOR_LAYER_SIZES[i+1]))

		self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[-1], self.num_shapes))

		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = nn.Softmax(dim=-1)

	def forward(self, X):
		for i in range(self.num_layers):
			fc = self.fc_list[i]
			X = fc(X)

			if i < self.num_layers - 1:
				X = self.LeReLU(X)
			else:
				X = self.activation(X)

		return X


# Predict probability distribution for boolean operation to apply
class OperationRegressor(nn.Module):
	def __init__(self, num_operations):
		super(OperationRegressor, self).__init__()
		self.num_operations = num_operations
		self.fc_list = nn.ModuleList()
		self.num_layers = len(REGRESSOR_LAYER_SIZES)

		for i in range(self.num_layers - 1):
			self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[i], REGRESSOR_LAYER_SIZES[i+1]))

		self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[-1], self.num_operations))

		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = nn.Softmax(dim=-1)

	def forward(self, X):
		for i in range(self.num_layers):
			fc = self.fc_list[i]
			X = fc(X)

			if i < self.num_layers - 1:
				X = self.LeReLU(X)
			else:
				X = self.activation(X)

		return X


# Predict 3D coordinate
class TranslationRegressor(nn.Module):
	def __init__(self, translation_scale):
		super(TranslationRegressor, self).__init__()
		self.translation_scale = translation_scale
		self.fc_list = nn.ModuleList()
		self.num_layers = len(REGRESSOR_LAYER_SIZES)
		self.output_size = 3

		for i in range(self.num_layers - 1):
			self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[i], REGRESSOR_LAYER_SIZES[i+1]))

		self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[-1], self.output_size))

		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = nn.Tanh()

	def forward(self, X):
		for i in range(self.num_layers):
			fc = self.fc_list[i]
			X = fc(X)

			if i < self.num_layers - 1:
				X = self.LeReLU(X)
			else:
				X = self.activation(X)

		# Restrict predicted coordinates to fit the output unit cube
		translation = X * self.translation_scale

		return translation


# Predict quaternion rotation
class RotationRegressor(nn.Module):
	def __init__(self):
		super(RotationRegressor, self).__init__()
		self.fc_list = nn.ModuleList()
		self.num_layers = len(REGRESSOR_LAYER_SIZES)
		self.output_size = 4

		for i in range(self.num_layers - 1):
			self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[i], REGRESSOR_LAYER_SIZES[i+1]))

		self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[-1], self.output_size))

		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)

	def forward(self, X):
		for i in range(self.num_layers):
			fc = self.fc_list[i]
			X = self.LeReLU(fc(X))

		# Normalize quaternion
		quaternion = torch.nn.functional.normalize(X, dim=-1)
		return quaternion


# Predict shape scale in x, y, and z axes
class ScaleRegressor(nn.Module):
	def __init__(self, min_scale, max_scale):
		super(ScaleRegressor, self).__init__()
		self.min_scale = min_scale
		self.max_scale = max_scale
		self.fc_list = nn.ModuleList()
		self.num_layers = len(REGRESSOR_LAYER_SIZES)
		self.output_size = 3

		for i in range(self.num_layers - 1):
			self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[i], REGRESSOR_LAYER_SIZES[i+1]))

		self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[-1], self.output_size))

		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = torch.sigmoid

	def forward(self, X):
		for i in range(self.num_layers):
			fc = self.fc_list[i]
			X = fc(X)

			if i < self.num_layers - 1:
				X = self.LeReLU(X)
			else:
				X = self.activation(X)

		# Restrict the predicted scale to expected range
		scale = (X * (self.max_scale - self.min_scale)) + self.min_scale

		return scale


# Predict amount to blend output primitive with current reconstruction
class BlendingRegressor(nn.Module):
	def __init__(self, min_blending, max_blending):
		super(BlendingRegressor, self).__init__()
		self.min_blending = min_blending
		self.max_blending = max_blending
		self.fc_list = nn.ModuleList()
		self.num_layers = len(REGRESSOR_LAYER_SIZES)
		self.output_size = 1

		for i in range(self.num_layers - 1):
			self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[i], REGRESSOR_LAYER_SIZES[i+1]))

		self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[-1], self.output_size))

		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = torch.sigmoid
	
	def forward(self, X):
		for i in range(self.num_layers):
			fc = self.fc_list[i]
			X = fc(X)

			if i < self.num_layers - 1:
				X = self.LeReLU(X)
			else:
				X = self.activation(X)

		# Restrict the blending factor to expected range
		blending = (X * (self.max_blending - self.min_blending)) + self.min_blending

		return blending


# Predict amount to round shape into sphere
class RoundnessRegressor(nn.Module):
	def __init__(self):
		super(RoundnessRegressor, self).__init__()
		self.fc_list = nn.ModuleList()
		self.num_layers = len(REGRESSOR_LAYER_SIZES)
		self.output_size = 1

		for i in range(self.num_layers - 1):
			self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[i], REGRESSOR_LAYER_SIZES[i+1]))

		self.fc_list.append(nn.Linear(REGRESSOR_LAYER_SIZES[-1], self.output_size))

		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = torch.sigmoid
	
	def forward(self, X):
		for i in range(self.num_layers):
			fc = self.fc_list[i]
			X = fc(X)

			if i < self.num_layers - 1:
				X = self.LeReLU(X)
			else:
				X = self.activation(X)

		return X


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

		self.shape = ShapeRegressor(num_shapes)
		self.operation = OperationRegressor(num_operations)
		self.translation = TranslationRegressor(translation_scale)
		self.rotation = RotationRegressor()
		self.scale = ScaleRegressor(min_scale, max_scale)
		self.blending = BlendingRegressor(min_blending, max_blending) if (predict_blending) else (None)
		self.roundness = RoundnessRegressor() if (predict_roundness) else (None)
	
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

	network = ShapeRegressor(3)
	outputs = network(inputs)
	print('Shape Output:')
	print(outputs, '\n')
	
	network = OperationRegressor(2)
	outputs = network(inputs)
	print('Boolean Regressor Output:')
	print(outputs, '\n')
	
	network = TranslationRegressor(0.6)
	outputs = network(inputs)
	print('Translation Output:')
	print(outputs, '\n')
	
	network = RotationRegressor()
	outputs = network(inputs)
	print('Rotation Output:')
	print(outputs, '\n')

	network = ScaleRegressor(0.005, 0.5)
	outputs = network(inputs)
	print('Scale Output:')
	print(outputs, '\n')

	network = BlendingRegressor(0, 1)
	outputs = network(inputs)
	print('Blending Output:')
	print(outputs, '\n')

	network = RoundnessRegressor()
	outputs = network(inputs)
	print('Roundness Output:')
	print(outputs, '\n')

	network = PrimitiveRegressor(3, 2)
	outputs = network(inputs)
	print('Combined Output Size:', len(outputs))