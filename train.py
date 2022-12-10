import os
import argparse
import torch
import signal
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform

from utilities.data_processing import *
from utilities.datasets import PointDataset

from networks.csg_crn import CSG_CRN
from utilities.sdf_csg import CSGModel
from losses.loss import Loss


# Percentage of data to use for training, validation, and testing
DATA_SPLIT = [0.85, 0.05, 0.1]
# Number of options for selecting primitives or operations
PRIMITIVES_SIZE = 3
OPERATIONS_SIZE = 2
# Weights for regressor functions
PRIMITIVE_WEIGHT = 0.01
SHAPE_WEIGHT = 0.01
OPERATION_WEIGHT = 0.01


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	# Data settings
	parser.add_argument('--data_dir', type=str, required=True, help='Dataset parent directory (data in subdirectories is included)')
	parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for checkpoints, trained model, and augmented dataset')
	parser.add_argument('--model_params', type=str, default='', help='Load model parameters from checkpoint file')
	parser.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')
	parser.add_argument('--no_preprocess', default=False, action='store_true', help='Disable near-surface sample preprocessing')
	parser.add_argument('--sample_dist', type=float, default=0.1, help='Distance from the surface to sample during preprocessing (Memory requirement increases for smaller sample_dist, must be >0)')

	# Model settings
	parser.add_argument('--num_input_points', type=int, default=1024, help='Number of points in the inputs (Memory requirement scales with num_input_points)')
	parser.add_argument('--num_loss_points', type=int, default=20000, help='Number of points to use when computing the loss')
	parser.add_argument('--num_prims', type=int, default=3, help='Number of primitives to generate before computing loss (Memory requirement scales with num_prims)')
	parser.add_argument('--num_iters', type=int, default=10, help='Number of refinement iterations to train for (Total generated primitives = num_prims x num_iters)')
	parser.add_argument('--clamp_dist', type=float, default=0.1, help='SDF clamping value for computing reconstruciton loss (Recommended to set clamp_dist to sample_dist)')
	parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size (Must be larger than 1)')
	parser.add_argument('--max_epochs', type=int, default=2000, help='Maximum number of epochs to train')

	# Training settings
	parser.add_argument('--checkpoint_freq', type=int, default=10, help='Number of epochs to train for before saving model parameters')
	parser.add_argument('--num_workers', type=int, default=8, help='Number of processes spawned for data loader')
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


# Prepare data files and load training dataset
def load_train_set(data_dir, output_dir, no_preprocess, sample_dist, num_input_points, num_loss_points, data_split):
	# Load sample files
	file_rel_paths = get_data_files(data_dir)
	print('Found %i data files' % len(file_rel_paths))

	# Create near-surface sample files
	if not no_preprocess:
		print('Selecting near-surface points...')
		data_dir = uniform_to_surface_data(data_dir, file_rel_paths, output_dir, sample_dist)

	# Split dataset and save to file
	(train_rel_paths, val_rel_paths, test_rel_paths) = torch.utils.data.random_split(file_rel_paths, data_split)
	save_list(os.path.join(output_dir, 'train.txt'), train_rel_paths)
	save_list(os.path.join(output_dir, 'val.txt'), val_rel_paths)
	save_list(os.path.join(output_dir, 'test.txt'), test_rel_paths)

	print('Iniitalizing dataset...')
	train_dataset = PointDataset(data_dir, train_rel_paths, num_input_points, num_loss_points)

	return (val_rel_paths, train_dataset)


# Load CSG-CRN network model
def load_model(primitives_size, operations_size, device, model_params=''):
	# Initialize model
	model = CSG_CRN(primitives_size, operations_size).to(device)

	# Load model parameters if available
	if model_params != '':
		model.load_state_dict(torch.load(model_params))

	return model


# Iteratively predict primitives and propagate average loss
def train_one_epoch(model, loss_func, optimizer, train_loader, sample_dist, num_prims, device, desc=''):
	total_loss = 0

	for (target_input_samples, target_all_samples) in tqdm(train_loader, desc=desc):
		# Load data
		target_all_points = target_all_samples[..., :3]
		target_all_distances = target_all_samples[..., 3]
		(batch_size, num_input_points, _) = target_input_samples.size()

		# Initialize SDF CSG model
		csg_model = CSGModel(device)

		# Send all data to training device
		target_all_points = target_all_points.to(device)
		target_all_distances = target_all_distances.to(device)
		target_input_samples = target_input_samples.to(device)

		# Sample initial reconstruction for loss function
		initial_loss_distances = csg_model.sample_csg(target_all_points)

		# Iteratively generate a set of primitives to build a CSG model
		for prim in range(num_prims):
			# Randomly sample initial reconstruction surface to generate input
			(initial_input_points, initial_input_distances) = csg_model.sample_csg_surface(batch_size, num_input_points, sample_dist)
			initial_input_samples = torch.cat((initial_input_points, initial_input_distances.unsqueeze(2)), dim=-1)
			# Predict next primitive
			outputs = model(target_input_samples, initial_input_samples)
			# Add primitive to CSG model
			csg_model.add_command(*outputs)

		# Sample generated CSG model
		refined_loss_distances = csg_model.sample_csg(target_all_points)

		# Get primitive shape and boolean operation propability distributions
		shapes_weights = torch.cat([x['shape weights'] for x in csg_model.csg_commands]).view(batch_size, num_prims, -1)
		operation_weights = torch.cat([x['operation weights'] for x in csg_model.csg_commands]).view(batch_size, num_prims, -1)

		# Compute loss
		batch_loss = loss_func(target_all_distances, initial_loss_distances, refined_loss_distances, shapes_weights, operation_weights)
		total_loss += batch_loss.item()

		# Back propagate
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()

	total_loss /= train_loader.__len__()
	return total_loss


# Train model for max_epochs or until stopped early
def train(model, loss_func, optimizer, train_loader, sample_dist, num_prims, checkpoint_dir, checkpoint_freq, device, max_epochs):
	model.train(True)

	for epoch in range(max_epochs):
		desc = f'Epoch {epoch+1}/{max_epochs}'
		loss = train_one_epoch(model, loss_func, optimizer, train_loader, sample_dist, num_prims, device, desc)
		print('Total Loss:', loss)

		# Save model parameters
		if (epoch+1) % checkpoint_freq == 0:
			checkpoint_path = os.path.join(checkpoint_dir, f'Epoch_{epoch+1}.pt')
			torch.save(model.state_dict(), checkpoint_path)


def main():
	args = options()
	print('')

	# Set training device
	device = get_device(args.device)
	torch.multiprocessing.set_start_method('spawn')

	# Initialize model
	model = load_model(PRIMITIVES_SIZE, OPERATIONS_SIZE, device, args.model_params)
	loss_func = Loss(args.clamp_dist, PRIMITIVE_WEIGHT, SHAPE_WEIGHT, OPERATION_WEIGHT).to(device)
	optimizer = torch.optim.Adam(model.parameters())

	# Load training set
	(output_dir, checkpoint_dir) = create_out_dir(args)
	(val_rel_paths, train_set) = load_train_set(args.data_dir, output_dir, args.no_preprocess, args.sample_dist, args.num_input_points, args.num_loss_points, DATA_SPLIT)
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

	# Train model
	print('')
	train(model, loss_func, optimizer, train_loader, args.sample_dist, args.num_prims, checkpoint_dir, args.checkpoint_freq, device, args.max_epochs)


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