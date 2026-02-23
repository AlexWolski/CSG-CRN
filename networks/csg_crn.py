import copy
import torch
import torch.nn as nn
from networks.pointnet import PointNetfeat, POINTNET_FEAT_OUTPUT_SIZE
from networks.regressor_decoder import PrimitiveRegressor
from utilities.constants import NEAR_SURFACE_SAMPLE_FACTOR, UNIFIED_SAMPLING
from utilities.csg_model import CSGModel, subtract_sdf, smooth_max
from utilities.sampler_utils import select_near_surface_samples


class CSG_CRN(nn.Module):
	def __init__(
			self, num_prims, num_shapes, num_operations, num_input_points, sample_dist, input_sampling_method, surface_uniform_ratio, device,
			decoder_layers=[], extended_input=False, predict_blending=True, predict_roundness=True, no_batch_norm=False):
		super(CSG_CRN, self).__init__()

		self.num_prims = num_prims
		self.num_shapes = num_shapes
		self.num_operations = num_operations
		self.num_input_points = num_input_points
		self.sample_dist = sample_dist
		self.input_sampling_method = input_sampling_method
		self.surface_uniform_ratio = surface_uniform_ratio
		self.device = device
		self.decoder_layers = decoder_layers
		self.extended_input = extended_input
		self.predict_blending = predict_blending
		self.predict_roundness = predict_roundness
		self.no_batch_norm = no_batch_norm

		num_feature_dims = 6 if self.extended_input else 4
		self.point_encoder = PointNetfeat(k=num_feature_dims, global_feat=True, input_transform=True, feature_transform=True, no_batch_norm=no_batch_norm)
		self.regressor_decoder_list = nn.ModuleList()

		# Initialize a separate decoder for each primitive
		for i in range(self.num_prims):
			regressor_decoder = PrimitiveRegressor(
				POINTNET_FEAT_OUTPUT_SIZE,
				self.num_shapes,
				self.num_operations,
				layer_sizes=self.decoder_layers,
				predict_blending=self.predict_blending,
				predict_roundness=self.predict_roundness,
				no_batch_norm=self.no_batch_norm
			)

			self.regressor_decoder_list.append(regressor_decoder)

		self.to(self.device)


	def forward(self, target_near_surface_samples, target_uniform_samples, csg_model=None):
		batch_size = target_near_surface_samples.size(dim=0)
		first_prim = csg_model is None

		if first_prim:
			csg_model = CSGModel(batch_size, device=self.device)

		# When using Unified sampling, generate near-surface samples by filtering by distance to both the target and reconstruction shapes.
		if self.input_sampling_method == UNIFIED_SAMPLING:
			num_near_surface_samples = target_near_surface_samples.size(1) // NEAR_SURFACE_SAMPLE_FACTOR
			target_near_surface_samples = select_near_surface_samples(target_near_surface_samples, num_near_surface_samples, csg_model)

		target_input_samples = self.combine_and_shuffle_samples(target_near_surface_samples, target_uniform_samples)
		input_tensor = self._get_input_tensor(target_input_samples, csg_model, first_prim, self.extended_input)

		# Change input shape from BxNxF to BxFxN for PointNet encoder
		# Where B = Batch Size, N = Number of Points, and F = Number of Features
		input_tensor = input_tensor.permute(0, 2, 1)

		# Encode target point cloud features
		features, _, _ = self.point_encoder(input_tensor)

		# Decode primitive predictions
		for decoder in self.regressor_decoder_list:
			csg_model.add_command(*decoder(features, first_prim))
			first_prim = False

		return csg_model


	# Create an input tensor with the correct format.
	def _get_input_tensor(self, target_input_samples, csg_model, first_prim, extended_input):
		# On the initial iteration, keep the target shape as the fill volume and append a dummy remove volume.
		if first_prim:
			return self._get_initial_input_tensor(target_input_samples, extended_input)
		# On refinement iterations, compute the volume to fill and remove to achieve the target shape.
		else:
			return self._get_refinement_input_tensor(target_input_samples, csg_model, extended_input)


	# Create an input tensor containing the target shape SDF with the correct format.
	def _get_initial_input_tensor(self, target_input_samples, extended_input):
		# Simplified input does not require dummy values.
		if not extended_input:
			return target_input_samples

		batch_size = target_input_samples.size(dim=0)
		num_points = target_input_samples.size(dim=1)

		null_volume_sdf = torch.ones((batch_size, num_points, 1), device=self.device)
		return torch.cat((target_input_samples, null_volume_sdf, null_volume_sdf), -1)


	# Create an input tensor containing the target and reconstruction shape SDFs with the correct format.
	def _get_refinement_input_tensor(self, target_input_samples, csg_model, extended_input):
		target_input_points = target_input_samples[:,:,:3]
		target_input_sdf = target_input_samples[:,:,3]

		# Sample the CSG model.
		init_recon_sdf = csg_model.sample_csg(target_input_points)
		# Volume of the target shape that still needs to be filled.
		missing_volume_sdf = subtract_sdf(target_input_sdf, init_recon_sdf).unsqueeze(-1)

		# Simplified input contains only missing volume SDF.
		if not extended_input:
			return torch.cat((target_input_points, missing_volume_sdf), -1)

		# Excess volume of the initial reconstruction that needs to be removed.
		excess_volume_sdf = subtract_sdf(init_recon_sdf, target_input_sdf).unsqueeze(-1)
		# Volume of the target shape that was correctly filled by the initial reconstruction.
		filled_volume_sdf = smooth_max(init_recon_sdf, target_input_sdf).unsqueeze(-1)
		# Append the fill and remove volumes to the sample points to create a unified input.
		return torch.cat((target_input_points, missing_volume_sdf, filled_volume_sdf, excess_volume_sdf), -1)


	def forward_cascade(self, target_near_surface_samples, target_uniform_samples, num_cascades):
		csg_model = self.forward(target_near_surface_samples, target_uniform_samples)

		for i in range(num_cascades):
			csg_model = self.forward(target_near_surface_samples, target_uniform_samples, csg_model)

		return csg_model


	# Train only the current cascade. Preivous cascades have trained parameters accessed through the `prev_cascades_list` parameter.
	def forward_separate_cascades(self, target_near_surface_samples, target_uniform_samples, prev_cascades_list):
		csg_model = None
		current_params = None
		is_training = self.training

		if len(prev_cascades_list) > 0:
			self.eval()
			current_params = copy.deepcopy(self.state_dict())

		# Generate previous cascasdes in inference mode. forward_separate_cascades will be called recursively on all cascades.
		for model_params in prev_cascades_list:
			with torch.no_grad():
				self.eval()
				self.load_state_dict(model_params)
				csg_model = self.forward(target_near_surface_samples, target_uniform_samples, csg_model).detach()

		# Generate current cascade
		if current_params != None:
			self.load_state_dict(current_params)

		# Run a forward pass on the current cascade
		self.train() if is_training else self.eval()
		return self.forward(target_near_surface_samples, target_uniform_samples, csg_model)


	# Set which operations to scale and by how much
	def set_operation_weight(self, scale_op, replace_op, operation_scale):
		for regressor_decoder in self.regressor_decoder_list:
			regressor_decoder.set_operation_weight(scale_op, replace_op, operation_scale)

	# Combine uniform and near-surface samples and shuffle results
	def combine_and_shuffle_samples(self, uniform_samples, near_surface_samples):
		input_samples = torch.cat((uniform_samples, near_surface_samples), 1)
		input_samples = input_samples[:, torch.randperm(input_samples.size(dim=1))]
		return input_samples


# Test network
def test():
	# Input Dimension: BxFxP
	# B = Batch Size
	# F = Feature Size
	# P = Point Size
	target_points = torch.rand(32, 1024, 4)
	initial_points = torch.rand(32, 1024, 4)

	csg_crn = CSG_CRN(3, 2)
	outputs = csg_crn(target_points, initial_points)
	print('Network Output Size:', len(outputs))
