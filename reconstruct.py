import signal
import argparse
import pyrender
import numpy as np
import torch

from networks.csg_crn import CSG_CRN
from utilities.csg_model import CSGModel


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_params', type=str, required=True, help='Load model parameters from file')
	parser.add_argument('--input_file', type=str, required=True, help='File containing sample points and SDF values of input shape')
	parser.add_argument('--num_input_points', type=int, required=True, help='Number of points to use from each input sample (Use same value as during training)')
	parser.add_argument('--num_prims', type=int, required=True, help='Number of primitives to generate before computing loss')
	parser.add_argument('--sample_dist', type=float, default=0.1, help='Distance from the surface to sample the reconstruction (Use same value as during training)')
	parser.add_argument('--sample_uniform', default=False, action='store_true', help='View generated reconstruciton with uniform samples instead of near-surface samples')
	parser.add_argument('--device', type=str, default='', help='Select preffered training device')

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
	# Load model parameters
	state_dict = torch.load(args.model_params)

	# Check for weights corresponding to blending and roundness regressors
	predict_blending = 'regressor_decoder.blending.fc1.weight' in state_dict
	predict_roundness = 'regressor_decoder.roundness.fc1.weight' in state_dict

	# Initialize model
	model = CSG_CRN(CSGModel.num_shapes, CSGModel.num_operations, predict_blending, predict_roundness).to(args.device)
	model.load_state_dict(state_dict)
	model.eval()

	return model


# Randomly sample input points
def load_input_points(args):
	# Load all points from file
	points = np.load(args.input_file).astype(np.float32)

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


def run_model(model, input_data, args):
	with torch.no_grad():
		# Initialize SDF CSG model
		csg_model = CSGModel(args.device)

		# Iteratively generate a set of primitives to build a CSG model
		for prim in range(args.num_prims):
			# Randomly sample initial reconstruction surface to generate input
			(initial_input_points, initial_input_distances) = csg_model.sample_csg_surface(1, args.num_input_points, args.sample_dist)
			initial_input_samples = torch.cat((initial_input_points, initial_input_distances.unsqueeze(2)), dim=-1)

			# Predict next primitive
			outputs = model(input_data, initial_input_samples)
			# Add primitive to CSG model
			csg_model.add_command(*outputs)

	return csg_model


def view_sdf(csg_model, num_points, point_size, args):
	if args.sample_uniform:
		(points, sdf) = csg_model.sample_csg_uniform(1, num_points)
	else:
		(points, sdf) = csg_model.sample_csg_surface(1, num_points, args.sample_dist)

	points = points[0].to(torch.device('cpu'))
	sdf = sdf[0].to(torch.device('cpu'))

	print(sdf)

	colors = np.zeros(points.shape)
	colors[sdf < 0, 2] = 1
	colors[sdf > 0, 0] = 1
	cloud = pyrender.Mesh.from_points(points, colors=colors)
	scene = pyrender.Scene()
	scene.add(cloud)
	viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=point_size)


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



def main():
	args = options()
	print('')

	# Run model
	args.device = get_device(args.device)
	model = load_model(args)
	input_data = load_input_points(args)
	csg_model = run_model(model, input_data, args)

	# Pretty print csg commands
	print_csg_commands(csg_model)

	# View reconstruction
	view_sdf(csg_model, 50000, 2, args)


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