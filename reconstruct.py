import os
import sys
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
from utilities.constants import SEPARATE_PARAMS
from utilities.csg_model import CSGModel, get_primitive_name, get_operation_name, add_sdf, subtract_sdf
from utilities.data_augmentation import RotationAxis
from utilities.sampler_utils import sample_from_mesh, sample_points_mesh_surface
from utilities.accuracy_metrics import compute_chamfer_distance
from utilities.csg_to_mesh import csg_to_mesh


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_params', type=str, required=True, help='Load model parameters from file.')
	parser.add_argument('--input_file', type=str, required=True, help='Model file to reconstruct.')
	parser.add_argument('--num_cascades', type=int, help='Number of refinement passes before back-propagating (Total generated primitives = num_prims * num_cascades)')
	parser.add_argument('--num_acc_points', type=int, default=30000, help='Number of points to use when computing accuracy.')
	parser.add_argument('--recon_resolution', type=int, default=256, help='Voxel resolution to use for the marching cubes algorithm when computing accuracy.')
	parser.add_argument('--num_view_points', type=int, default=10000, help='Number of points to visualize the output.')
	parser.add_argument('--point_size', type=int, default=3, help='Size to render each point of the point cloud.')
	parser.add_argument('--device', type=str, default='', help='Select preferred inference device')

	args = parser.parse_args()
	return args


# Determine device to train on
def get_device(device=None):
	if device:
		return torch.device(device)
	elif torch.cuda.is_available():
		return torch.device('cuda')
	else:
		return torch.device('cpu')


def load_model(args):
	# Load model parameters and arguments
	torch.serialization.add_safe_globals([argparse.Namespace, Subset, RotationAxis, timedelta])
	save_data = torch.load(args.model_params, weights_only=True)
	state_dict = save_data['model']
	saved_args = save_data['args']
	prev_cascades_list = save_data['prev_cascades_list']

	# Load training args
	args.num_input_points = saved_args.num_input_points
	args.sample_dist = saved_args.sample_dist
	args.surface_uniform_ratio = saved_args.surface_uniform_ratio
	args.loss_metric = saved_args.loss_metric
	args.clamp_dist = saved_args.clamp_dist
	args.sub_weight = saved_args.sub_weight
	args.cascade_training_mode = saved_args.cascade_training_mode

	if args.num_cascades == None:
		args.num_cascades = saved_args.num_cascades

	predict_blending = not saved_args.no_blending
	predict_roundness = not saved_args.no_roundness

	# Initialize model
	model = CSG_CRN(
		saved_args.num_prims,
		CSGModel.num_shapes,
		CSGModel.num_operations,
		args.num_input_points,
		args.sample_dist,
		saved_args.surface_uniform_ratio,
		args.device,
		saved_args.decoder_layers,
		not saved_args.no_extended_input,
		predict_blending,
		predict_roundness,
		saved_args.no_batch_norm
	)

	model.load_state_dict(state_dict, strict=False)
	model.set_operation_weight(subtract_sdf, add_sdf, args.sub_weight)
	model.eval()

	return (model, prev_cascades_list)


# Load sample points from file
def load_mesh_and_samples(input_file, args):
	mesh = trimesh.load(input_file)
	mesh = scale_to_unit_sphere(mesh)

	num_uniform_samples = math.ceil(args.num_input_points * args.surface_uniform_ratio)
	num_surface_samples = math.floor(args.num_input_points * (1 - args.surface_uniform_ratio))

	# Compute samples
	(
		uniform_points, uniform_distances,
		near_surface_points, near_surface_distances,
		surface_points
	) = sample_from_mesh(mesh, num_uniform_samples, args.num_acc_points, num_surface_samples, args.sample_dist)

	# Combine samples
	uniform_samples = torch.cat((uniform_points, uniform_distances.unsqueeze(-1)), dim=-1).unsqueeze(0).to(args.device)
	near_surface_samples = torch.cat((near_surface_points, near_surface_distances.unsqueeze(-1)), dim=-1).unsqueeze(0).to(args.device)
	surface_points = surface_points.unsqueeze(0).to(args.device)

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


def print_recon_loss(near_surface_samples, uniform_samples, surface_points, csg_model, loss_metric):
	recon_loss = ReconstructionLoss(loss_metric)
	print(f'Reconstruction {loss_metric} Loss:')
	print(recon_loss.forward(near_surface_samples, uniform_samples, surface_points, csg_model))
	print('')


def print_chamfer_dist(target_mesh, recon_mesh, num_acc_points, device):
	target_points = sample_points_mesh_surface(target_mesh, num_acc_points).unsqueeze(0).to(device)
	recon_points = sample_points_mesh_surface(recon_mesh, num_acc_points).unsqueeze(0).to(device)
	accuracy = compute_chamfer_distance(target_points, recon_points, no_grad=True)
	print('Chamfer Distance:')
	print(accuracy)
	print('')


def construct_csg_model(model, input_file, args, prev_cascades_list=None):
	target_mesh, uniform_samples, near_surface_samples, surface_points = load_mesh_and_samples(input_file, args)

	if args.cascade_training_mode == SEPARATE_PARAMS:
		csg_model = model.forward_separate_cascades(uniform_samples, near_surface_samples, prev_cascades_list)
	else:
		csg_model = model.forward_cascade(uniform_samples, near_surface_samples, args.num_cascades)

	recon_mesh = csg_to_mesh(csg_model, args.recon_resolution)[0]

	# Pretty print csg commands
	print_csg_commands(csg_model)
	# Print reconstruction loss
	print_recon_loss(near_surface_samples, uniform_samples, surface_points, csg_model, args.loss_metric)
	# Print reconstruction accuracy
	print_chamfer_dist(target_mesh, recon_mesh, args.num_acc_points, args.device)

	return (target_mesh, recon_mesh, csg_model)


def main():
	args = options()
	print('')

	# Run model
	args.device = get_device(args.device)
	(model, prev_cascades_list) = load_model(args)

	# View reconstruction
	get_mesh_and_csg_model = lambda input_file: construct_csg_model(model, input_file, args, prev_cascades_list)
	window_title = "Reconstruct: " + os.path.basename(args.input_file)

	try:
		viewer = SdfModelViewer("Reconstructed SDF", args.point_size, args.num_view_points, args.input_file, args.sample_dist, get_mesh_and_csg_model)
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
