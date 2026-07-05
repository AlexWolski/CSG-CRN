import torch
import torch.nn as nn
from utilities.csg_model import add_sdf, CSGModel


# Tune Leaky ReLU slope for predicting negative values
LEAKY_RELU_NEGATIVE_SLOPE = 0.2

# Default settings for regressor decoder
DEFAULT_TRANSLATION_SCALE = 1
DEFAULT_MIN_SCALE = 0.005
DEFAULT_MAX_SCALE = 1
DEFAULT_MIN_BLENDING = 0.001
DEFAULT_MAX_BLENDING = 1


# Simple fully-connected MLP network with optional batch-normalization and Leaky-Relu activation.
class MLPNetwork(nn.Module):
	def __init__(self, layer_sizes, no_batch_norm=False):
		super(MLPNetwork, self).__init__()
		self.fc_list = nn.ModuleList()
		self.bn_list = nn.ModuleList() if not no_batch_norm else None
		self.LeReLU = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, True)
		self._init_fc_layers(layer_sizes)


	def _init_fc_layers(self, layer_sizes):
		for i in range(len(layer_sizes) - 1):
			prev_layer_size = layer_sizes[i]
			curr_layer_size = layer_sizes[i+1]
			self.fc_list.append(nn.Linear(prev_layer_size, curr_layer_size))

			if self.bn_list != None and i+1 < len(layer_sizes) - 1:
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
		
		return X


# Parent regressor network class to generalize network building
class RegressorNetwork(nn.Module):
	def __init__(self, layer_sizes, activ_func=None, activ_func_args=None, norm_func=None, no_batch_norm=False):
		super(RegressorNetwork, self).__init__()
		self.activ_func = activ_func
		self.activ_func_args = activ_func_args
		self.norm_func = norm_func
		self.fc_network = MLPNetwork(layer_sizes, no_batch_norm=no_batch_norm)


	def forward(self, X):
		# Forward MLP layers.
		X = self.fc_network.forward(X)

		# Apply activation function.
		if self.activ_func != None:
			if self.activ_func_args:
				X = self.activ_func(X, **self.activ_func_args)
			else:
				X = self.activ_func(X)

		# Apply regressor normalization function.
		if self.norm_func != None:
			X = self.norm_func(X)

		return X


# Predict all primitive parameters 
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
		prim_decoder_layer_sizes=[],
		regressor_layer_sizes=[],
		translation_scale=DEFAULT_TRANSLATION_SCALE,
		min_scale=DEFAULT_MIN_SCALE,
		max_scale=DEFAULT_MAX_SCALE,
		predict_blending=True,
		min_blending=DEFAULT_MIN_BLENDING,
		max_blending=DEFAULT_MAX_BLENDING,
		predict_roundness=True,
		no_batch_norm=False):

		super(PrimitiveRegressor, self).__init__()

		gumbel_softmax_args = {'hard': True, 'tau': 0.5, 'dim': -1}
		prim_decoder_layer_sizes = [input_feature_size] + prim_decoder_layer_sizes
		regressor_layer_sizes = [prim_decoder_layer_sizes[-1]] + regressor_layer_sizes

		# Primitive feature vector decoder network.
		self.prim_feat_network = MLPNetwork(prim_decoder_layer_sizes, no_batch_norm=no_batch_norm)

		# Primitive attribute regressor networks.
		self.shape = RegressorNetwork(regressor_layer_sizes + [num_shapes], activ_func=nn.functional.gumbel_softmax, activ_func_args=gumbel_softmax_args, no_batch_norm=no_batch_norm)
		self.operation = RegressorNetwork(regressor_layer_sizes + [num_operations], activ_func=nn.functional.gumbel_softmax, activ_func_args=gumbel_softmax_args, no_batch_norm=no_batch_norm)
		self.translation = RegressorNetwork(regressor_layer_sizes + [3], activ_func=nn.Tanh(), norm_func=self._normalizeTranslation(translation_scale), no_batch_norm=no_batch_norm)
		self.rotation = RegressorNetwork(regressor_layer_sizes + [4], activ_func=None, norm_func=self._normalizeRotation(), no_batch_norm=no_batch_norm)
		self.scale = RegressorNetwork(regressor_layer_sizes + [3], activ_func=torch.sigmoid, norm_func=self._normalizeScale(min_scale, max_scale), no_batch_norm=no_batch_norm)
		self.blending = RegressorNetwork(regressor_layer_sizes + [1], activ_func=torch.sigmoid, norm_func=self._normalizeBlending(min_blending, max_blending), no_batch_norm=no_batch_norm) if (predict_blending) else (None)
		self.roundness = RegressorNetwork(regressor_layer_sizes + [1], activ_func=torch.sigmoid, no_batch_norm=no_batch_norm) if (predict_roundness) else (None)

		self.scale_op_id = None
		self.replace_op_id = None
		self.operation_weight = None


	# Set which operations to scale and by how much
	def set_operation_weight(self, scale_op, replace_op, operation_weight):
		self.scale_op_id = CSGModel.operation_functions.index(scale_op) if scale_op in CSGModel.operation_functions else None
		self.replace_op_id = CSGModel.operation_functions.index(replace_op) if replace_op in CSGModel.operation_functions else None
		self.operation_weight = operation_weight


	# Generate an operation tensor of size (BxO) where B is the batch size and O is the number of operations supported by the CSG model.
	def get_add_op_tensor(self, batch_size, device):
		add_op_index = CSGModel.operation_functions.index(add_sdf)
		add_op_tensor = torch.zeros((batch_size, CSGModel.num_operations), device=device)
		add_op_tensor[:,add_op_index] = 1
		return add_op_tensor


	# Scale operation vector
	def scale_operations(self, operation):
		if self.scale_op_id is None or self.replace_op_id is None or self.operation_weight is None:
			return operation

		operation[:, self.replace_op_id] += operation[:, self.scale_op_id] * (1 - self.operation_weight)
		operation[:, self.scale_op_id] *= self.operation_weight

		return operation


	def forward(self, X, first_prim):
		batch_size = X.size(dim=0)

		# Forward MLP layers.
		X = self.prim_feat_network.forward(X)

		# Decode primitive attributes.
		shape = self.shape.forward(X)
		operation = self.operation.forward(X) if not first_prim else self.get_add_op_tensor(batch_size, X.get_device())
		translation = self.translation.forward(X)
		rotation = self.rotation.forward(X)
		scale = self.scale.forward(X)
		blending = self.blending.forward(X) if self.blending is not None and not first_prim else None
		roundness = self.roundness.forward(X) if self.roundness is not None else None

		# Scale operation vector
		operation = self.scale_operations(operation)

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

	network = PrimitiveRegressor([64], 3, 2)
	outputs = network(inputs)
	print('Combined Output Size:', len(outputs))
