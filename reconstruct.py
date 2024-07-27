import os
import signal
import argparse
import pyrender
import numpy as np
import torch

from networks.csg_crn import CSG_CRN
from utilities.csg_model import CSGModel
from losses.reconstruction_loss import ReconstructionLoss


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_params', type=str, required=True, help='Load model parameters from file')
	parser.add_argument('--input_file', type=str, required=True, help='File containing sample points and SDF values of input shape')
	parser.add_argument('--num_prims', type=int, required=True, help='Number of primitives to generate')
	parser.add_argument('--view_sampling', default='near-surface', choices=['uniform', 'near-surface'], nargs=1, help='Visualize uniform SDF samples or samples near recosntruction surface')
	parser.add_argument('--num_view_points', type=int, default=100000, help='Number of points to visualize the output')
	parser.add_argument('--show_exterior_points', default=False, action='store_true', help='Show points outside of the represented shape')
	parser.add_argument('--device', type=str, default='', help='Select preferred training device')

	args = parser.parse_args()
	return args


# Determine device to train on
def get_device(device):
	if device:
		device = torch.device(device)

	elif torch.cuda.is_available():
		device = torch.device('cuda')

	else:
		device = torch.device('cpu')

	return device


def load_model(args):
	# Load model parameters and arguments
	save_data = torch.load(args.model_params)
	state_dict = save_data['model']
	saved_args = save_data['args']

	# Load training args
	args.num_input_points = saved_args.num_input_points
	args.sample_method = saved_args.sample_method
	args.sample_dist = saved_args.sample_dist

	# Check for weights corresponding to blending and roundness regressors
	predict_blending = 'regressor_decoder.blending.fc1.weight' in state_dict
	predict_roundness = 'regressor_decoder.roundness.fc1.weight' in state_dict
	no_batch_norm = not 'point_encoder.bn1.weight' in state_dict

	# Initialize model
	model = CSG_CRN(CSGModel.num_shapes, CSGModel.num_operations, predict_blending, predict_roundness, no_batch_norm).to(args.device)
	model.load_state_dict(state_dict)
	model.eval()

	return model


# Randomly sample input points
def load_input_samples(args):
	# Load all points from file
	points = np.load(args.input_file).astype(np.float32)

	# Select near-surface points if needed
	if args.sample_method[0] == 'near-surface':
		surface_sample_rows = np.where(abs(points[:,3]) <= args.sample_dist)
		points = points[surface_sample_rows]

	# Randomly select needed number of input surface points
	replace = (points.shape[0] < args.num_input_points)
	select_rows = np.random.choice(points.shape[0], args.num_input_points, replace=replace)
	select_input_points = points[select_rows]

	# Convert to torch tensor
	select_input_points = torch.from_numpy(select_input_points)

	# Set batch size to 1
	select_input_points = select_input_points.unsqueeze(0)

	# Send data to compute device
	select_input_points = select_input_points.to(args.device)
	
	return select_input_points


def run_model(model, input_samples, args):
	with torch.no_grad():
		# Initialize SDF CSG model
		csg_model = CSGModel(args.device)

		# Iteratively generate a set of primitives to build a CSG model
		for prim in range(args.num_prims):
			# Randomly sample initial reconstruction surface to generate input
			if args.sample_method[0] == 'uniform':
				initial_input_samples = csg_model.sample_csg_uniform(1, args.num_input_points)
			else:
				initial_input_samples = csg_model.sample_csg_surface(1, args.num_input_points, args.sample_dist)

			if initial_input_samples is not None:
				(initial_input_points, initial_input_distances) = initial_input_samples
				initial_input_samples = torch.cat((initial_input_points, initial_input_distances.unsqueeze(2)), dim=-1)

			# Predict next primitive
			outputs = model(input_samples, initial_input_samples)
			# Add primitive to CSG model
			csg_model.add_command(*outputs)

	return csg_model


def view_sdf(csg_model, num_points, point_size, args):
	filename = os.path.basename(args.input_file)

	if args.view_sampling[0] == 'uniform':
		(points, sdf) = csg_model.sample_csg_uniform(1, num_points)
	else:
		(points, sdf) = csg_model.sample_csg_surface(1, num_points, args.sample_dist)

	if not args.show_exterior_points:
		points = points[sdf <= 0, :]
		sdf = sdf[sdf <= 0]

	points = points.to(torch.device('cpu'))
	sdf = sdf.to(torch.device('cpu'))

	colors = np.zeros(points.shape)
	colors[sdf < 0, 2] = 1
	colors[sdf > 0, 0] = 1
	cloud = pyrender.Mesh.from_points(points, colors=colors)
	scene = pyrender.Scene()
	scene.add(cloud)
	viewer = pyrender.Viewer(scene,
		use_raymond_lighting=True,
		point_size=point_size,
		show_world_axis=True,
		viewport_size=(1000,1000),
		window_title="Reconstruct: " + filename,
		view_center=[0,0,0])


def pretty_print_tensor(message, tensor):
	print(message, end='')

	raw_list = tensor.tolist()[0]
	pretty_list = [f'{item:.5f}' for item in raw_list]

	print(pretty_list)


def print_csg_commands(csg_model):
	count = 1

	for command in csg_model.csg_commands:
		print(f'Command {count}:')
		pretty_print_tensor('Shape:\t\t', command['shape weights'])
		pretty_print_tensor('Operation:\t', command['operation weights'])
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


def main():
	args = options()
	print('')

	# Run model
	args.device = get_device(args.device)
	model = load_model(args)
	input_samples = load_input_samples(args)
	csg_model = run_model(model, input_samples, args)

	# Pretty print csg commands
	print_csg_commands(csg_model)

	# Print reconstruction loss
	print_recon_loss(input_samples, csg_model, args)

	# View reconstruction
	view_sdf(csg_model, args.num_view_points, 2, args)


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