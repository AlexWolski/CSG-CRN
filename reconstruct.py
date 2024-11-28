import os
import signal
import argparse
import pyrender
import numpy as np
import torch

from networks.csg_crn import CSG_CRN
from utilities.csg_model import CSGModel, get_primitive_name, get_operation_name
from losses.reconstruction_loss import ReconstructionLoss
from view_sdf import SdfModelViewer
from utilities.file_loader import FileLoader


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_params', type=str, required=True, help='Load model parameters from file')
	parser.add_argument('--input_file', type=str, required=True, help='Numpy file containing sample points and SDF values of input shape')
	parser.add_argument('--num_view_points', type=int, default=100000, help='Number of points to visualize the output')
	parser.add_argument('--show_exterior_points', default=False, action='store_true', help='Show points outside of the represented shape')
	parser.add_argument('--point_size', type=int, default=2, help='Size to render each point of the point cloud')

	args = parser.parse_args()
	return args


# Determine device to train on
def get_device():
	return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');


def load_model(args):
	# Load model parameters and arguments
	save_data = torch.load(args.model_params)
	state_dict = save_data['model']
	saved_args = save_data['args']

	# Load training args
	args.num_input_points = saved_args.num_input_points
	args.num_prims = saved_args.num_prims
	args.sample_method = saved_args.sample_method
	args.sample_dist = saved_args.sample_dist
	args.no_blending = saved_args.no_blending
	args.no_roundness = saved_args.no_roundness
	args.no_batch_norm = saved_args.no_batch_norm

	predict_blending = not args.no_blending
	predict_roundness = not args.no_roundness

	# Initialize model
	model = CSG_CRN(args.num_prims, CSGModel.num_shapes, CSGModel.num_operations, predict_blending, predict_roundness, args.no_batch_norm).to(args.device)
	model.load_state_dict(state_dict, strict=False)
	model.eval()

	return model


# Randomly sample input points
def load_input_samples(input_file, args):
	# Load all points from file
	points = np.load(input_file).astype(np.float32)

	# Select near-surface points if needed
	if args.sample_method[0] == 'near-surface':
		surface_sample_rows = np.where(abs(points[:,3]) <= args.sample_dist)
		points = points[surface_sample_rows]

	# Randomly select needed number of input surface points
	replace = (points.shape[0] < args.num_input_points)
	select_rows = np.random.choice(points.shape[0], args.num_input_points, replace=replace)
	select_input_points = points[select_rows]

	# Convert to torch tensor
	select_input_points = torch.from_numpy(select_input_points).to(args.device).unsqueeze(0)
	
	return select_input_points


def run_model(model, input_samples, args):
	with torch.no_grad():
		# Initialize SDF CSG model
		csg_model = CSGModel(args.device)

		# Randomly sample initial reconstruction surface to generate input
		if args.sample_method[0] == 'uniform':
			initial_input_samples = csg_model.sample_csg_uniform(1, args.num_input_points)
		else:
			initial_input_samples = csg_model.sample_csg_surface(1, args.num_input_points, args.sample_dist)

		if initial_input_samples is not None:
			(initial_input_points, initial_input_distances) = initial_input_samples
			initial_input_samples = torch.cat((initial_input_points, initial_input_distances.unsqueeze(2)), dim=-1)

		# Predict next primitive
		output_list = model(input_samples, initial_input_samples)

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
		pretty_print_tensor('Translation:\t', command['transforms'][0])
		pretty_print_tensor('Rotation:\t', command['transforms'][1])
		pretty_print_tensor('Scale:\t\t', command['transforms'][2])

		if command['blending'] is not None:
			pretty_print_tensor('Blending:\t', command['blending'])

		if command['roundness'] is not None:
			pretty_print_tensor('Roundness:\t', command['roundness'])

		print('')
		count += 1


def print_recon_loss(input_samples, csg_model, args):
	input_points = input_samples[:,:,:3]
	input_sdf = input_samples[:,:,3]

	csg_sdf = csg_model.sample_csg(input_points)

	recon_loss = ReconstructionLoss()
	print('Reconstruction Loss:')
	print(recon_loss.forward(input_sdf, csg_sdf))


def construct_csg_model(model, input_file, args):
	input_samples = load_input_samples(input_file, args)
	csg_model = run_model(model, input_samples, args)

	# Pretty print csg commands
	print_csg_commands(csg_model)
	# Print reconstruction loss
	print_recon_loss(input_samples, csg_model, args)
	print('\n')

	return csg_model


def main():
	args = options()
	print('')

	# Run model
	args.device = get_device()
	model = load_model(args)
	csg_model = construct_csg_model(model, args.input_file, args)

	# View reconstruction
	get_csg_model = lambda input_file: construct_csg_model(model, input_file, args)
	window_title = "Reconstruct: " + os.path.basename(args.input_file)
	SdfModelViewer("Reconstructed SDF", args.point_size, False, args.num_view_points, args.input_file, csg_model, args.sample_dist, get_csg_model)


if __name__ == '__main__':
	# Catch CTRL+Z force shutdown
	def exit_handler(signum, frame):
		print('\nClearing GPU cache')
		torch.cuda.empty_cache()
		print('Enter CTRL+C multiple times to exit')
		sys.exit()

	signal.signal(signal.SIGTSTP, exit_handler)

	# Catch CTRL+C force shutdown
	try:
		main()
	except KeyboardInterrupt:
		print('\nClearing GPU cache and quitting')
		torch.cuda.empty_cache()