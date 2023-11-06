import os
import argparse
import yaml
import torch
import signal
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform
from torch.optim import Adam, lr_scheduler

from utilities.data_processing import *
from utilities.datasets import PointDataset

from networks.csg_crn import CSG_CRN
from utilities.csg_model import CSGModel
from losses.loss import Loss


# Percentage of data to use for training, validation, and testing
DATA_SPLIT = [0.85, 0.05, 0.1]
# Weights for regressor functions
PRIM_LOSS_WEIGHT = 0.01
SHAPE_LOSS_WEIGHT = 0.01
OP_LOSS_WEIGHT = 0.01


# Parse commandline arguments
def options():
	parser1 = argparse.ArgumentParser()
	parser2 = argparse.ArgumentParser()

	# Data settings
	parser1.add_argument('--data_dir', type=str, required=True, help='Dataset parent directory (data in subdirectories is included)')
	parser1.add_argument('--output_dir', type=str, default='./output', help='Output directory for checkpoints, trained model, and augmented dataset')
	parser1.add_argument('--model_path', type=str, default='', help='Load parameters and settings from saved model file. Overwrites all other model settings')
	parser1.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')

	args, remaining_args = parser1.parse_known_args()

	# Load settings from saved model file
	if args.model_path != '':
		data_dir = args.data_dir
		output_dir = args.output_dir
		overwrite = args.overwrite

		save_data = torch.load(args.model_path)
		args = save_data['args']

		args.data_dir = data_dir
		args.output_dir = output_dir
		args.overwrite = overwrite

	# Parse all arguments
	else:
		# Model settings
		parser2.add_argument('--num_input_points', type=int, default=1024, help='Number of points to use from each input sample (Memory requirement scales linearly with num_input_points)')
		parser2.add_argument('--num_loss_points', type=int, default=20000, help='Number of points to use when computing the loss')
		parser2.add_argument('--num_prims', type=int, default=3, help='Number of primitives to generate before computing loss (Memory requirement scales with num_prims)')
		parser2.add_argument('--num_iters', type=int, default=10, help='Number of refinement iterations to train for (Total generated primitives = num_prims x num_iters)')
		parser2.add_argument('--no_blending', default=False, action='store_true', help='Disable primitive blending')
		parser2.add_argument('--no_roundness', default=False, action='store_true', help='Disable primitive rounding')
		parser2.add_argument('--no_batch_norm', default=False, action='store_true', help='Disable batch normalization')
		parser2.add_argument('--sample_method', default=['uniform'], choices=['uniform', 'near-surface'], nargs=1, help='Select SDF samples uniformly or near object surfaces. Near-surface requires pre-processing')
		parser2.add_argument('--sample_dist', type=float, default=0.1, help='Maximum distance to object surface for near-surface sampling (Smaller sample_dist increases memory requirement, must be >0)')
		parser2.add_argument('--clamp_dist', type=float, default=0.1, help='SDF clamping value for computing reconstruciton loss (Recommended to set clamp_dist to sample_dist)')

		# Training settings
		parser2.add_argument('--batch_size', type=int, default=32, help='Mini-batch size. When set to 1, batch normalization is disabled')
		parser2.add_argument('--keep_last_batch', default=False, action='store_true', help='Train on remaining data samples at the end of each epoch')
		parser2.add_argument('--max_epochs', type=int, default=2000, help='Maximum number of epochs to train')
		parser2.add_argument('--lr_patience', type=int, default=5, help='Number of training epochs without improvement before the learning rate is adjusted')
		parser2.add_argument('--early_stop_patience', type=int, default=10, help='Number of training epochs without improvement before training terminates')
		parser2.add_argument('--checkpoint_freq', type=int, default=10, help='Number of epochs to train for before saving model parameters')
		parser2.add_argument('--device', type=str, default='', help='Select preferred training device')

		args = parser2.parse_args(args=remaining_args, namespace=args)

	# Expand paths
	args.data_dir = os.path.abspath(args.data_dir)

	if args.model_path != '':
		args.model_path = os.path.abspath(args.model_path)

	if args.output_dir != '':
		args.output_dir = os.path.abspath(args.output_dir)

	# Disable batch norm for SDG
	if args.batch_size == 1:
		args.no_batch_norm = True

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
def load_data_sets(args, data_split):
	# Load sample files
	file_rel_paths = get_data_files(args.data_dir)
	print('Found %i data files' % len(file_rel_paths))

	# Split dataset
	(train_split, val_split, test_split) = torch.utils.data.random_split(file_rel_paths, data_split)

	# Ensure each dataset has enough samples
	for dataset in [('Train', train_split), ('Validation', val_split), ('Test', test_split)]:
		# Check if any dataset is empty
		if len(dataset[1].indices) == 0:
			err_msg = f'{dataset[0]} dataset is empty! Add more data samples'
			raise Exception(err_msg)

		# Check if batch size is larger than dataset size
		if (not args.keep_last_batch) and (len(dataset[1].indices) < args.batch_size):
			err_msg = f'{dataset[0]} dataset ({len(dataset[1].indices)}) is smaller than batch size ({args.batch_size})! Add data samples or set keep_last_batch option'
			raise Exception(err_msg)

	# Create near-surface sample files
	if args.sample_method[0] == 'near-surface':
		print('Selecting near-surface points...')
		args.data_dir = uniform_to_surface_data(args, file_rel_paths)

	# Save dataset lists
	save_list(os.path.join(args.output_dir, 'train.txt'), train_split)
	save_list(os.path.join(args.output_dir, 'val.txt'), val_split)
	save_list(os.path.join(args.output_dir, 'test.txt'), test_split)

	print(f'Training set:\t{len(train_split.indices)} samples')
	print(f'Validation set:\t{len(val_split.indices)} samples')
	print(f'Testing set:\t{len(test_split.indices)} samples')

	train_dataset = PointDataset(args.data_dir, train_split, args.num_input_points, args.num_loss_points)
	val_dataset = PointDataset(args.data_dir, val_split, args.num_input_points, args.num_loss_points)

	return (train_dataset, val_dataset)


# Load CSG-CRN network model
def load_model(num_shapes, num_operations, args, device):
	predict_blending = not args.no_blending
	predict_roundness = not args.no_roundness

	# Initialize model
	model = CSG_CRN(num_shapes, num_operations, predict_blending, predict_roundness, args.no_batch_norm).to(device)

	# Load model parameters if available
	if args.model_path != '':
		model.load_state_dict(torch.load(args.model_path))

	return model

# Run a forwards pass of the network model
def model_forward(model, loss_func, target_input_samples, target_all_samples, args, device):
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
	for prim in range(args.num_prims):
		# Randomly sample initial reconstruction surface to generate input
		if args.sample_method[0] == 'uniform':
			(initial_input_points, initial_input_distances) = csg_model.sample_csg_uniform(batch_size, num_input_points)
		else:
			(initial_input_points, initial_input_distances) = csg_model.sample_csg_surface(batch_size, num_input_points, args.sample_dist)

		initial_input_samples = torch.cat((initial_input_points, initial_input_distances.unsqueeze(2)), dim=-1)
		# Predict next primitive
		outputs = model(target_input_samples, initial_input_samples)
		# Add primitive to CSG model
		csg_model.add_command(*outputs)

	# Sample generated CSG model
	refined_loss_distances = csg_model.sample_csg(target_all_points)

	# Get primitive shape and boolean operation propability distributions
	shapes_weights = torch.cat([x['shape weights'] for x in csg_model.csg_commands]).view(batch_size, args.num_prims, -1)
	operation_weights = torch.cat([x['operation weights'] for x in csg_model.csg_commands]).view(batch_size, args.num_prims, -1)

	# Compute loss
	loss = loss_func(target_all_distances, initial_loss_distances, refined_loss_distances, shapes_weights, operation_weights)

	return loss


# Iteratively predict primitives and propagate average loss
def train_one_epoch(model, loss_func, optimizer, train_loader, args, device, desc=''):
	total_train_loss = 0

	for (target_input_samples, target_all_samples) in tqdm(train_loader, desc=desc):
		# Forward pass
		batch_loss = model_forward(model, loss_func, target_input_samples, target_all_samples, args, device)
		total_train_loss += batch_loss.item()

		# Back propagate
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()

	total_train_loss /= train_loader.__len__()
	return total_train_loss


def validate(model, loss_func, val_loader, args, device):
	total_val_loss = 0

	with torch.no_grad():
		for (target_input_samples, target_all_samples) in val_loader:
			batch_loss = model_forward(model, loss_func, target_input_samples, target_all_samples, args, device)
			total_val_loss += batch_loss.item()

	total_val_loss /= val_loader.__len__()
	return total_val_loss


# Train model for max_epochs or until stopped early
def train(model, loss_func, optimizer, scheduler, train_loader, val_loader, args, device):
	model.train(True)

	early_stop_counter = 0
	min_val_loss = float('inf')

	# Train until model stops improving or a maximum number of epochs is reached
	for epoch in range(args.max_epochs):
		# Train model
		desc = f'Epoch {epoch+1}/{args.max_epochs}'
		train_loss = train_one_epoch(model, loss_func, optimizer, train_loader, args, device, desc)
		val_loss = validate(model, loss_func, val_loader, args, device)
		scheduler.step(val_loss)

		print('Training Loss:  ', train_loss)
		print('Validation Loss:', val_loss)
		print('Learning Rate:  ', optimizer.param_groups[0]['lr'])

		# Check for early stopping
		if val_loss < min_val_loss:
			min_val_loss = val_loss
			early_stop_counter = 0

			# Save final trained model
			trained_model_path = os.path.join(args.output_dir, 'best_model.pt')
			torch.save({'model': model.state_dict(), 'args': args}, trained_model_path)
		else:
			early_stop_counter += 1
			
		if early_stop_counter > args.early_stop_patience:
			print(f'Stopping Training. Validation loss has not improved in {args.early_stop_patience} epochs')
			break

		# Save checkpoint parameters
		if (epoch+1) % args.checkpoint_freq == 0:
			checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pt')
			torch.save({'model': model.state_dict(), 'args': args}, checkpoint_path)
			print(f'Checkpoint saved to:')
			print(checkpoint_path)

	print('\nTraining complete! Model parameters saved to:')
	print(trained_model_path)


def main():
	args = options()
	print('')

	# Set training device
	device = get_device(args.device)

	# Initialize model
	model = load_model(CSGModel.num_shapes, CSGModel.num_operations, args, device)
	loss_func = Loss(args.clamp_dist, PRIM_LOSS_WEIGHT, SHAPE_LOSS_WEIGHT, OP_LOSS_WEIGHT).to(device)
	optimizer = Adam(model.parameters())
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience)

	# Load training set
	(args.output_dir, args.checkpoint_dir) = create_out_dir(args)
	(train_dataset, val_dataset) = load_data_sets(args, DATA_SPLIT)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last= not args.keep_last_batch)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last= not args.keep_last_batch)

	# Save settings to file
	settings_path = os.path.join(args.output_dir, 'settings.yml')

	with open(settings_path, 'w') as out_path:
		yaml.dump(args.__dict__, out_path, sort_keys=False)

	# Train model
	print('')
	train(model, loss_func, optimizer, scheduler, train_loader, val_loader, args, device)


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