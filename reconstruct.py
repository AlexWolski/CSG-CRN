import os
import sys
import copy
import math
import signal
import argparse
import trimesh
import traceback
import torch
import time

from datetime import timedelta
from torch.utils.data import Subset
from networks.csg_crn import CSG_CRN
from mesh_to_sdf.utils import scale_to_unit_sphere
from losses.reconstruction_loss import ReconstructionLoss
from view_sdf import SdfModelViewer
from utilities.constants import SEPARATE_PARAMS, INIT_RECON
from utilities.csg_model import CSGModel, get_primitive_name, get_operation_name, add_sdf, subtract_sdf
from utilities.data_augmentation import RotationAxis
from utilities.device_utils import get_device
from utilities.sampler_utils import sample_from_mesh, sample_points_mesh_surface
from utilities.accuracy_metrics import compute_chamfer_distance
from utilities.csg_to_mesh import csg_to_mesh


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_params', type=str, required=True, help='Load model parameters from file.')
	parser.add_argument('--input_file', type=str, required=True, help='Model file to reconstruct.')
	parser.add_argument('--num_cascades', type=int, help='Number of refinement passes before back-propagating. Total generated primitives = num_prims + (num_prims * num_cascades)')
	parser.add_argument('--num_acc_points', type=int, default=30000, help='Number of points to use when computing accuracy.')
	parser.add_argument('--recon_resolution', type=int, default=256, help='Voxel resolution to use for the marching cubes algorithm when computing accuracy.')
	parser.add_argument('--num_view_points', type=int, default=10000, help='Number of points to visualize the output.')
	parser.add_argument('--point_size', type=int, default=3, help='Size to render each point of the point cloud.')
	parser.add_argument('--device', type=str, default=[], nargs='*', help='Select preferred inference device. Reconstruction only supports a single device')

	args = parser.parse_args()
	return args

# Initialize model from parameters file.
def load_model(model_params, device, num_cascades=None):
	# Load model parameters and arguments
	torch.serialization.add_safe_globals([argparse.Namespace, Subset, RotationAxis, timedelta])
	save_data = torch.load(model_params, weights_only=True, map_location=device)
	state_dict = save_data['model']
	saved_args = save_data['args']
	init_model_state_dict = save_data['init_model']
	prev_cascades_list = save_data['prev_cascades_list']

	# Overwrite num_cascades when provided
	if num_cascades is not None:
		saved_args.num_cascades = num_cascades

	# Fallback for new arguments
	saved_args.excess_loss_weight = getattr(saved_args, 'excess_loss_weight', None)
	saved_args.residual_only_training = getattr(saved_args, 'residual_only_training', False)

	predict_blending = not saved_args.no_blending
	predict_roundness = not saved_args.no_roundness

	# Initialize model
	model = CSG_CRN(
		saved_args.num_prims,
		CSGModel.num_shapes,
		CSGModel.num_operations,
		saved_args.num_input_points,
		saved_args.sample_dist,
		saved_args.input_sampling_method,
		saved_args.surface_uniform_ratio,
		device,
		saved_args.encoder_layers,
		saved_args.encoder_trans_conv_layers,
		saved_args.encoder_trans_fc_layers,
		saved_args.prim_decoder_layers,
		saved_args.regressor_layers,
		not saved_args.no_extended_input,
		predict_blending,
		predict_roundness,
		not saved_args.no_extended_pooling,
		saved_args.no_batch_norm,
		residual_only_training=saved_args.residual_only_training
	)

	model.load_state_dict(state_dict, strict=False)
	model.set_operation_weight(subtract_sdf, add_sdf, saved_args.sub_weight)
	model.eval()

	return (model, saved_args, init_model_state_dict, prev_cascades_list)


# Load sample points from file
def load_mesh_and_samples(input_file, saved_args, num_acc_points, device):
	mesh = trimesh.load(input_file)
	mesh = scale_to_unit_sphere(mesh)

	num_uniform_samples = math.ceil(saved_args.num_input_points * saved_args.surface_uniform_ratio)
	num_surface_samples = math.floor(saved_args.num_input_points * (1 - saved_args.surface_uniform_ratio))

	# Compute samples
	(
		uniform_points, uniform_distances,
		near_surface_points, near_surface_distances,
		surface_points
	) = sample_from_mesh(mesh, num_uniform_samples, num_acc_points, num_surface_samples, saved_args.sample_dist)

	# Combine samples
	uniform_samples = torch.cat((uniform_points, uniform_distances.unsqueeze(-1)), dim=-1).unsqueeze(0).to(device)
	near_surface_samples = torch.cat((near_surface_points, near_surface_distances.unsqueeze(-1)), dim=-1).unsqueeze(0).to(device)
	surface_points = surface_points.unsqueeze(0).to(device)

	return (mesh, uniform_samples, near_surface_samples, surface_points)


# Randomly sample input points
def combine_samples(uniform_samples, near_surface_samples, num_input_points):
	input_samples = torch.cat((uniform_samples, near_surface_samples), dim=1)

	# Shuffle data samples
	input_samples = input_samples[:, torch.randperm(num_input_points)]

	# Add batch dimension
	return input_samples


def pretty_print_tensor(message, tensor):
	print(message, end='')

	raw_list = tensor.tolist()[0]
	pretty_list = [f'{item:.5f}' for item in raw_list]

	print(pretty_list)


def print_csg_commands(csg_model):
	count = 1

	for command in csg_model.csg_commands:
		shape_weights = command['shape weights']
		operation_weights = command['operation weights']

		print(f'Command {count}:')
		print(f'Shape:\t\t{get_primitive_name(shape_weights)}')
		print(f'Operation:\t{get_operation_name(operation_weights)}\t[{operation_weights.tolist()}]')
		pretty_print_tensor('Translation:\t', command['translations'])
		pretty_print_tensor('Rotation:\t', command['rotations'])
		pretty_print_tensor('Scale:\t\t', command['scales'])

		if command['blending'] is not None:
			pretty_print_tensor('Blending:\t', command['blending'])

		if command['roundness'] is not None:
			pretty_print_tensor('Roundness:\t', command['roundness'])

		print('')
		count += 1


def compute_recon_loss(near_surface_samples, uniform_samples, surface_points, csg_model, loss_metric, excess_loss_weight):
	recon_loss = ReconstructionLoss(loss_metric, excess_loss_weight)
	return recon_loss.forward(near_surface_samples, uniform_samples, surface_points, csg_model)


def compute_chamfer_distance_mesh(target_mesh, recon_mesh, num_acc_points, device):
	target_points = sample_points_mesh_surface(target_mesh, num_acc_points).unsqueeze(0).to(device)
	recon_points = sample_points_mesh_surface(recon_mesh, num_acc_points).unsqueeze(0).to(device)
	return compute_chamfer_distance(target_points, recon_points, no_grad=True)


def model_inference(model, saved_args, init_model_state_dict, prev_cascades_list, near_surface_samples, uniform_samples):
	csg_model = None

	if saved_args.cascade_training_mode == INIT_RECON:
		# Run a forward pass on the inital reconstruciton model.
		current_state_dict = copy.deepcopy(model.state_dict())
		model.load_state_dict(init_model_state_dict, strict=False)
		csg_model = model.forward(near_surface_samples, uniform_samples)
		# Revert the CSGCRN model to the reconstruction weights.
		model.load_state_dict(current_state_dict)

	if saved_args.cascade_training_mode == SEPARATE_PARAMS:
		return model.forward_separate_cascades(near_surface_samples, uniform_samples, prev_cascades_list)
	else:
		return model.forward_cascade(near_surface_samples, uniform_samples, saved_args.num_cascades, csg_model)


def construct_csg_model(model, input_file, args, saved_args, device, init_model_state_dict=None, prev_cascades_list=None):
	target_mesh, uniform_samples, near_surface_samples, surface_points = load_mesh_and_samples(input_file, saved_args, args.num_acc_points, device)
	csg_model = model_inference(model, saved_args, init_model_state_dict, prev_cascades_list, near_surface_samples, uniform_samples)
	recon_mesh = csg_to_mesh(csg_model, args.recon_resolution)[0]

	# Pretty print csg commands
	print_csg_commands(csg_model)

	# Print reconstruction loss
	print(f'Reconstruction {saved_args.loss_metric} Loss:')
	print(compute_recon_loss(near_surface_samples, uniform_samples, surface_points, csg_model, saved_args.loss_metric, saved_args.excess_loss_weight))
	print('')

	# Print reconstruction accuracy
	print('Chamfer Distance:')
	print(compute_chamfer_distance_mesh(target_mesh, recon_mesh, args.num_acc_points, device))
	print('')

	return (target_mesh, recon_mesh, csg_model)


def get_inference_device(device):
	# Assert a single inference device.
	if len(device) > 1:
		print('Reconstruction only supports one device for inference. Select one device or select "None" to automatically select one.')
		exit()

	if len(device) > 0:
		device = device[0]

	return get_device(device, cpu_allowed=True)


def main():
	args = options()
	print('')

	device = get_inference_device(args.device)

	# Run model
	(model, saved_args, init_model_state_dict, prev_cascades_list) = load_model(args.model_params, device, args.num_cascades)

	# View reconstruction
	get_mesh_and_csg_model = lambda input_file: construct_csg_model(model, input_file, args, saved_args, device, init_model_state_dict, prev_cascades_list)
	_window_title = "Reconstruct: " + os.path.basename(args.input_file)

	try:
		viewer = SdfModelViewer("Reconstructed SDF", args.point_size, args.num_view_points, args.input_file, saved_args.sample_dist, get_mesh_and_csg_model)
		await_viewer(viewer)
	except FileNotFoundError as fileError:
		print(fileError)
	except Exception:
		print(traceback.format_exc())


# Wait for the viewer to be closed
def await_viewer(viewer):
	# Catch CTRL+Z force shutdown
	signal.signal(signal.SIGTSTP, lambda _signum, _frame: exit_handler(viewer))

	# Wait for the viewer to be closed
	try:
		while viewer.is_active:
			time.sleep(0.1)
	# Catch CTRL+C force shutdown
	except KeyboardInterrupt:
		print('\nProgram interrupted by keyboard input')
	finally:
		exit_handler(viewer)


# Gracefully close the external viewer and exit the program
def exit_handler(viewer):
	viewer.close_external()

	while viewer.is_active:
		time.sleep(0.1)

	print('\nClearing GPU cache and quitting')
	torch.cuda.empty_cache()
	sys.exit()


if __name__ == '__main__':
	main()
