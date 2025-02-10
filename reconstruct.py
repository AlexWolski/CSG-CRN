import os
import sys
import math
import signal
import argparse
import trimesh
import traceback
import mesh_to_sdf
import pyrender
import numpy as np
import torch
import time

from torch.utils.data import Subset
from networks.csg_crn import CSG_CRN
from mesh_to_sdf.utils import scale_to_unit_sphere
from losses.reconstruction_loss import ReconstructionLoss
from view_sdf import SdfModelViewer
from utilities.csg_model import CSGModel, get_primitive_name, get_operation_name
from utilities.file_loader import FileLoader
from utilities.data_augmentation import RotationAxis
from utilities.sampler_utils import sample_from_mesh, sample_points_mesh_surface, sample_csg_surface
from utilities.accuracy_metrics import compute_chamfer_distance
from utilities.csg_to_mesh import csg_to_mesh


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_params', type=str, required=True, help='Load model parameters from file.')
	parser.add_argument('--input_file', type=str, required=True, help='Model file to reconstruct.')
	parser.add_argument('--num_acc_points', type=int, default=30000, help='Number of points to use when computing validation accuracy.')
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
	torch.serialization.add_safe_globals([argparse.Namespace, Subset, RotationAxis])
	save_data = torch.load(args.model_params, weights_only=True)
	state_dict = save_data['model']
	saved_args = save_data['args']

	# Load training args
	args.num_input_points = saved_args.num_input_points
	args.num_prims = saved_args.num_prims
	args.surface_uniform_ratio = saved_args.surface_uniform_ratio
	args.sample_dist = saved_args.sample_dist
	args.decoder_layers = saved_args.decoder_layers
	args.no_blending = saved_args.no_blending
	args.no_roundness = saved_args.no_roundness
	args.no_batch_norm = saved_args.no_batch_norm

	predict_blending = not args.no_blending
	predict_roundness = not args.no_roundness

	# Initialize model
	model = CSG_CRN(args.num_prims, CSGModel.num_shapes, CSGModel.num_operations, args.decoder_layers, predict_blending, predict_roundness, args.no_batch_norm).to(args.device)
	model.load_state_dict(state_dict, strict=False)
	model.eval()

	return model


# Randomly sample input points
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
	uniform_samples = torch.cat((uniform_points, uniform_distances.unsqueeze(-1)), dim=-1)
	surface_samples = torch.cat((near_surface_points, near_surface_distances.unsqueeze(-1)), dim=-1)
	input_samples = torch.cat((uniform_samples, surface_samples))

	# Shuffle data samples
	input_samples = input_samples[torch.randperm(args.num_input_points)]

	# Add batch dimension
	return (mesh, input_samples.unsqueeze(0).to(args.device))


# TODO: Add support for iterative generation
def run_model(model, input_samples, args):
	with torch.no_grad():
		# Initialize SDF CSG model
		csg_model = CSGModel(args.device)

		# Predict next primitive
		output_list = model(input_samples, None)

		# Add primitives to CSG model
		for output in output_list:
			csg_model.add_command(*output)

	return csg_model


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
		print(f'Operation:\t{get_operation_name(operation_weights)}')
		pretty_print_tensor('Translation:\t', command['translations'])
		pretty_print_tensor('Rotation:\t', command['rotations'])
		pretty_print_tensor('Scale:\t\t', command['scales'])

		if command['blending'] is not None:
			pretty_print_tensor('Blending:\t', command['blending'])

		if command['roundness'] is not None:
			pretty_print_tensor('Roundness:\t', command['roundness'])

		print('')
		count += 1


def print_recon_loss(input_samples, csg_model):
	input_points = input_samples[:,:,:3]
	input_sdf = input_samples[:,:,3]

	csg_sdf = csg_model.sample_csg(input_points)

	recon_loss = ReconstructionLoss()
	print('Reconstruction Loss:')
	print(recon_loss.forward(input_sdf, csg_sdf))
	print('')


def print_chamfer_dist(target_mesh, recon_mesh, num_acc_points, device):
	target_points = sample_points_mesh_surface(target_mesh, num_acc_points).unsqueeze(0).to(device)
	recon_points = sample_points_mesh_surface(recon_mesh, num_acc_points).unsqueeze(0).to(device)
	accuracy = compute_chamfer_distance(target_points, recon_points)
	print('Chamfer Distance:')
	print(accuracy)
	print('')


def construct_csg_model(model, input_file, args):
	target_mesh, input_samples = load_mesh_and_samples(input_file, args)
	csg_model = run_model(model, input_samples, args)
	recon_mesh = csg_to_mesh(csg_model, args.recon_resolution)[0]

	# Pretty print csg commands
	print_csg_commands(csg_model)
	# Print reconstruction loss
	print_recon_loss(input_samples, csg_model)
	# Print reconstruction accuracy
	print_chamfer_dist(target_mesh, recon_mesh, args.num_acc_points, args.device)

	return (target_mesh, recon_mesh, csg_model)


def main():
	args = options()
	print('')

	# Run model
	args.device = get_device(args.device)
	model = load_model(args)

	# View reconstruction
	get_mesh_and_csg_model = lambda input_file: construct_csg_model(model, input_file, args)
	window_title = "Reconstruct: " + os.path.basename(args.input_file)

	try:
		viewer = SdfModelViewer("Reconstructed SDF", args.point_size, False, args.num_view_points, args.input_file, args.sample_dist, get_mesh_and_csg_model)
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
