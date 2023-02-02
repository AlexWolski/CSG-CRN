import torch
import torch.nn as nn


# Dimensions of the layers in each regressor
REGRESSOR_LAYER_SIZE = 256
# Tune Leaky ReLU slope for predicting negative values
LEAKY_RELU_NEGATIVE_SLOPE = 0.2


# Predict probability distribution for output shape primitive
class ShapeRegressor(nn.Module):
	def __init__(self, num_shapes):
		super(ShapeRegressor, self).__init__()
		self.num_shapes = num_shapes

		self.fc1 = nn.Linear(REGRESSOR_LAYER_SIZE, REGRESSOR_LAYER_SIZE)
		self.fc2 = nn.Linear(REGRESSOR_LAYER_SIZE, num_shapes)
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = nn.Softmax(dim=-1)

	def forward(self, X):
		X = self.LeReLU(self.fc1(X))
		shape = self.activation(self.fc2(X))

		return shape


# Predict probability distribution for boolean operation to apply
class OperationRegressor(nn.Module):
	def __init__(self, num_operations):
		super(OperationRegressor, self).__init__()
		self.num_operations = num_operations

		self.fc1 = nn.Linear(REGRESSOR_LAYER_SIZE, REGRESSOR_LAYER_SIZE)
		self.fc2 = nn.Linear(REGRESSOR_LAYER_SIZE, num_operations)
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = nn.Softmax(dim=-1)

	def forward(self, X):
		X = self.LeReLU(self.fc1(X))
		operation = self.activation(self.fc2(X))

		return operation


# Predict 3D coordinate
class TranslationRegressor(nn.Module):
	def __init__(self, translation_scale):
		super(TranslationRegressor, self).__init__()
		self.translation_scale = translation_scale

		self.fc1 = nn.Linear(REGRESSOR_LAYER_SIZE, REGRESSOR_LAYER_SIZE)
		self.fc2 = nn.Linear(REGRESSOR_LAYER_SIZE, 3)
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = nn.Tanh()

	def forward(self, X):
		X = self.LeReLU(self.fc1(X))
		translation = self.activation(self.fc2(X))

		# Restrict predicted coordinates to fit the output unit cube
		translation = translation * self.translation_scale

		return translation


# Predict quaternion rotation
class RotationRegressor(nn.Module):
	def __init__(self):
		super(RotationRegressor, self).__init__()
		self.fc1 = nn.Linear(REGRESSOR_LAYER_SIZE, REGRESSOR_LAYER_SIZE)
		self.fc2 = nn.Linear(REGRESSOR_LAYER_SIZE, 4)
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)

	def forward(self, X):
		X = self.LeReLU(self.fc1(X))
		quaternion = self.LeReLU(self.fc2(X))

		# Normalize quaternion
		quaternion = torch.nn.functional.normalize(quaternion, dim=-1)

		return quaternion


# Predict shape scale in x, y, and z axes
class ScaleRegressor(nn.Module):
	def __init__(self, min_scale, max_scale):
		super(ScaleRegressor, self).__init__()
		self.min_scale = min_scale
		self.max_scale = max_scale

		self.fc1 = nn.Linear(REGRESSOR_LAYER_SIZE, REGRESSOR_LAYER_SIZE)
		self.fc2 = nn.Linear(REGRESSOR_LAYER_SIZE, 3)
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = torch.sigmoid

	def forward(self, X):
		X = self.LeReLU(self.fc1(X))
		scale = self.activation(self.fc2(X))

		# Restrict the predicted scale to expected range
		scale = (scale * (self.max_scale - self.min_scale)) + self.min_scale

		return scale


# Predict amount to blend output primitive with current reconstruction
class BlendingRegressor(nn.Module):
	def __init__(self, min_blending, max_blending):
		super(BlendingRegressor, self).__init__()
		self.min_blending = min_blending
		self.max_blending = max_blending

		self.fc1 = nn.Linear(REGRESSOR_LAYER_SIZE, REGRESSOR_LAYER_SIZE)
		self.fc2 = nn.Linear(REGRESSOR_LAYER_SIZE, 1)
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = torch.sigmoid
	
	def forward(self, X):
		X = self.LeReLU(self.fc1(X))
		blending = self.activation(self.fc2(X))

		# Restrict the blending factor to expected range
		blending = (blending * (self.max_blending - self.min_blending)) + self.min_blending

		return blending


# Predict amount to round shape into sphere
class RoundnessRegressor(nn.Module):
	def __init__(self):
		super(RoundnessRegressor, self).__init__()

		self.fc1 = nn.Linear(REGRESSOR_LAYER_SIZE, REGRESSOR_LAYER_SIZE)
		self.fc2 = nn.Linear(REGRESSOR_LAYER_SIZE, 1)
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self.activation = torch.sigmoid
	
	def forward(self, X):
		X = self.LeReLU(self.fc1(X))
		roundness = self.activation(self.fc2(X))

		return roundness


# Pridict all primitive parameters 
class PrimitiveRegressor(nn.Module):
	def __init__(self,
		num_shapes, num_operations,
		translation_scale=0.6,
		min_scale=0.005, max_scale=0.5,
		predict_blending=True,
		min_blending=0.001, max_blending=1,
		predict_roundness=True):

		super(PrimitiveRegressor, self).__init__()

		self.shape = ShapeRegressor(num_shapes)
		self.operation = OperationRegressor(num_operations)
		self.translation = TranslationRegressor(translation_scale)
		self.rotation = RotationRegressor()
		self.scale = ScaleRegressor(min_scale, max_scale)
		self.blending = BlendingRegressor(min_blending, max_blending) if (predict_blending) else (None)
		self.roundness = RoundnessRegressor() if (predict_roundness) else (None)
	
	def forward(self, X):
		shape = self.shape.forward(X)
		operation = self.operation.forward(X)
		translation = self.translation.forward(X)
		rotation = self.rotation.forward(X)
		scale = self.scale.forward(X)
		blending = self.blending.forward(X) if (self.blending is not None) else (None)
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