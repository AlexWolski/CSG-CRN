import os
import argparse
import yaml
import torch
import traceback
import signal
import sys
from tqdm import tqdm
from torch import autocast

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.distributions.uniform import Uniform
from torch.optim import AdamW, lr_scheduler

from utilities.data_processing import *
from utilities.datasets import PointDataset
from utilities.data_augmentation import get_augment_parser
from utilities.early_stopping import EarlyStopping

from networks.csg_crn import CSG_CRN
from utilities.csg_model import CSGModel
from losses.loss import Loss


# Percentage of data to use for training, validation, and testing
DATA_SPLIT = [0.8, 0.1, 0.1]
# Weights for regressor functions
PRIM_LOSS_WEIGHT = 0.01
SHAPE_LOSS_WEIGHT = 0.01
OP_LOSS_WEIGHT = 0.01


# Parse commandline arguments
def options():
	help_parser = get_help_parser()
	data_parser = get_data_parser()
	model_parser = get_model_parser()
	augment_parser = get_augment_parser('ONLINE AUGMENT SETTINGS')


	# Parse and handle Help argument
	args, remaining_args = help_parser.parse_known_args()

	if args.help or not remaining_args:
		print()
		data_parser.print_help()
		print('\n')
		model_parser.print_help()
		print('\n')
		augment_parser.print_help()
		exit()

	# Parse data settings
	args, remaining_args = data_parser.parse_known_args(args=remaining_args, namespace=args)

	# Load settings from saved model file
	if args.model_path:
		print('\nLoading Arguments From Model File:')
		print(os.path.abspath(args.model_path))

		# Cache data settings
		data_args = (args.model_path, args.data_dir, args.output_dir, args.overwrite, args.skip_preprocess)
		# Load arguments from modle file
		args = torch.load(args.model_path)['args']
		# Apply data settings
		(args.model_path, args.data_dir, args.output_dir, args.overwrite, args.skip_preprocess) = data_args

	# Parse all arguments
	else:
		args, remaining_args = model_parser.parse_known_args(args=remaining_args, namespace=args)
		augment_parser.parse_args(args=remaining_args, namespace=args)


	# Expand paths
	args.data_dir = os.path.abspath(args.data_dir)
	args.model_path = os.path.abspath(args.model_path) if args.model_path else None
	args.output_dir = os.path.abspath(args.output_dir)

	# Disable batch norm for SGD
	args.no_batch_norm = True if args.batch_size == 1 else args.no_batch_norm

    # Print arguments
	print('\nArguments:')
	print('----------')

	for arg in vars(args):
		print('{0:20} - {1}'.format(arg, getattr(args, arg)))

	return args


def get_help_parser():
	help_parser = argparse.ArgumentParser(add_help=False)
	help_parser.add_argument('-h', '--help', default=False, action='store_true', help='Print help text')

	return help_parser


def get_data_parser():
	data_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	data_group = data_parser.add_argument_group('DATA SETTINGS')

	data_group.add_argument('--data_dir', type=str, required=True, help='[REQUIRED] Dataset parent directory (data in subdirectories is included)')
	data_group.add_argument('--output_dir', type=str, default='./output', help='Output directory for checkpoints, trained model, and augmented dataset')
	data_group.add_argument('--model_path', type=str, default='', help='Load parameters and settings from saved model file. Overwrites all other model settings')
	data_group.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')
	data_group.add_argument('--skip_preprocess', default=False, action='store_true', help='Skip the pre-processing step if the provided data_dir already contains samples of the proper length and sampling method')

	return data_parser


def get_model_parser():
	model_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	model_group = model_parser.add_argument_group('MODEL SETTINGS')
	training_group = model_parser.add_argument_group('TRAINING SETTINGS')
	
	# Model settings
	model_group.add_argument('--num_input_points', type=int, default=1024, help='Number of points to use from each input sample (Memory requirement scales linearly with num_input_points)')
	model_group.add_argument('--num_loss_points', type=int, default=20000, help='Number of points to use when computing the loss')
	model_group.add_argument('--num_prims', type=int, default=3, help='Number of primitives to generate before computing loss (Memory requirement scales with num_prims)')
	model_group.add_argument('--num_iters', type=int, default=10, help='Number of refinement iterations to train for (Total generated primitives = num_prims x num_iters)')
	model_group.add_argument('--no_blending', default=False, action='store_true', help='Disable primitive blending')
	model_group.add_argument('--no_roundness', default=False, action='store_true', help='Disable primitive rounding')
	model_group.add_argument('--no_batch_norm', default=False, action='store_true', help='Disable batch normalization')
	model_group.add_argument('--sample_method', default=['near-surface'], choices=['uniform', 'near-surface'], nargs=1, help='Select SDF samples uniformly or near object surfaces. Near-surface requires pre-processing')
	model_group.add_argument('--sample_dist', type=float, default=0.1, help='Maximum distance to object surface for near-surface sampling (must be >0)')

	# Training settings
	training_group.add_argument('--batch_size', type=int, default=32, help='Mini-batch size. When set to 1, batch normalization is disabled')
	training_group.add_argument('--keep_last_batch', default=False, action='store_true', help='Train on remaining data samples at the end of each epoch')
	training_group.add_argument('--max_epochs', type=int, default=2000, help='Maximum number of epochs to train')
	training_group.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate')
	training_group.add_argument('--lr_factor', type=float, default=0.1, help='Learning rate reduction factor')
	training_group.add_argument('--lr_patience', type=int, default=10, help='Number of training epochs without improvement before the learning rate is adjusted')
	training_group.add_argument('--lr_threshold', type=float, default=0.05, help='Minimum recognized percentage of improvement over previous loss')
	training_group.add_argument('--early_stop_patience', type=int, default=20, help='Number of training epochs without improvement before training terminates')
	training_group.add_argument('--early_stop_threshold', type=float, default=0.05, help='Minimum recognized percentage of improvement over previous loss')
	training_group.add_argument('--checkpoint_freq', type=int, default=10, help='Number of epochs to train for before saving model parameters')
	training_group.add_argument('--device', type=str, default='', help='Select preferred training device')
	training_group.add_argument('--disable_amp', default=False, action='store_true', help='Disable Automatic Mixed Precision')

	return model_parser


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
def load_data_splits(args, data_split, device):
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

		num_samples = len(dataset[1].indices)
		num_augment_samples = num_samples * args.augment_copies

		# Check if batch size is larger than dataset size
		if not args.keep_last_batch and num_augment_samples < args.batch_size:
			err_msg = f'{dataset[0]} dataset ({num_augment_samples}) is smaller than batch size ({args.batch_size})! Add data samples or set keep_last_batch option'
			raise Exception(err_msg)

	# Save dataset lists
	save_list(os.path.join(args.output_dir, 'train.txt'), train_split)
	save_list(os.path.join(args.output_dir, 'val.txt'), val_split)
	save_list(os.path.join(args.output_dir, 'test.txt'), test_split)

	print(f'Training set:\t{len(train_split.indices)} samples')
	print(f'Validation set:\t{len(val_split.indices)} samples')
	print(f'Testing set:\t{len(test_split.indices)} samples\n')

	return (train_split, val_split, test_split)


# Load CSG-CRN network model
def load_model(num_shapes, num_operations, args, device):
	predict_blending = not args.no_blending
	predict_roundness = not args.no_roundness

	# Initialize model
	model = CSG_CRN(num_shapes, num_operations, predict_blending, predict_roundness, args.no_batch_norm).to(device)

	# Load model parameters if available
	if args.model_path:
		model.load_state_dict(torch.load(args.model_path)['model'])

	return model


# Run a forwards pass of the network model
def model_forward(model, loss_func, target_input_samples, target_loss_samples, args, device):
	# Load data
	target_all_points = target_loss_samples[..., :3]
	target_loss_distances = target_loss_samples[..., 3]
	(batch_size, num_input_points, _) = target_input_samples.size()

	# Initialize SDF CSG model
	csg_model = CSGModel(device)

	# Sample initial reconstruction for loss function
	initial_loss_distances = csg_model.sample_csg(target_all_points)

	# Iteratively generate a set of primitives to build a CSG model
	for prim in range(args.num_prims):
		# Randomly sample initial reconstruction surface to generate input
		if args.sample_method[0] == 'uniform':
			initial_input_samples = csg_model.sample_csg_uniform(batch_size, num_input_points)
		else:
			initial_input_samples = csg_model.sample_csg_surface(batch_size, num_input_points, args.sample_dist)

		if initial_input_samples is not None:
			(initial_input_points, initial_input_distances) = initial_input_samples
			initial_input_samples = torch.cat((initial_input_points, initial_input_distances.unsqueeze(2)), dim=-1)

		# Predict next primitive
		with autocast(device_type=device.type, dtype=torch.float16, enabled=not args.disable_amp):
			outputs = model(target_input_samples, initial_input_samples)

		# Add primitive to CSG model
		csg_model.add_command(*outputs)

	# Sample generated CSG model
	refined_loss_distances = csg_model.sample_csg(target_all_points)

	# Get primitive shape and boolean operation propability distributions
	shapes_weights = torch.cat([x['shape weights'] for x in csg_model.csg_commands]).view(batch_size, args.num_prims, -1)
	operation_weights = torch.cat([x['operation weights'] for x in csg_model.csg_commands]).view(batch_size, args.num_prims, -1)

	# Compute loss
	loss = loss_func(target_loss_distances, refined_loss_distances, shapes_weights, operation_weights)

	return loss


# Iteratively predict primitives and propagate average loss
def train_one_epoch(model, loss_func, optimizer, scaler, train_loader, args, device, desc=''):
	total_train_loss = 0

	for (target_input_samples, target_loss_samples) in tqdm(train_loader, desc=desc):
		target_input_samples = target_input_samples.squeeze(0)
		target_loss_samples = target_loss_samples.squeeze(0)

		# Forward pass
		batch_loss = model_forward(model, loss_func, target_input_samples, target_loss_samples, args, device)
		total_train_loss += batch_loss.item()

		# Back propagate
		optimizer.zero_grad(set_to_none=True)
		scaler.scale(batch_loss).backward()
		scaler.step(optimizer)
		scaler.update()

	total_train_loss /= train_loader.__len__()
	return total_train_loss


def validate(model, loss_func, val_loader, args, device):
	total_val_loss = 0

	with torch.no_grad():
		for (target_input_samples, target_loss_samples) in val_loader:
			target_input_samples = target_input_samples.squeeze(0)
			target_loss_samples = target_loss_samples.squeeze(0)

			batch_loss = model_forward(model, loss_func, target_input_samples, target_loss_samples, args, device)
			total_val_loss += batch_loss.item()

	total_val_loss /= val_loader.__len__()
	return total_val_loss


# Train model for max_epochs or until stopped early
def train(model, loss_func, optimizer, scheduler, scaler, train_loader, val_loader, args, device):
	model.train(True)

	# Initialize early stopper
	trained_model_path = os.path.join(args.output_dir, 'best_model.pt')
	save_best_model = lambda _: torch.save({'model': model.state_dict(), 'args': args}, trained_model_path)
	early_stopping = EarlyStopping(args.early_stop_patience, args.early_stop_threshold, save_best_model)

	# Train until model stops improving or a maximum number of epochs is reached
	for epoch in range(args.max_epochs):
		# Train model
		desc = f'Epoch {epoch+1}/{args.max_epochs}'
		train_loss = train_one_epoch(model, loss_func, optimizer, scaler, train_loader, args, device, desc)
		val_loss = validate(model, loss_func, val_loader, args, device)
		scheduler.step(val_loss)
		early_stopping(val_loss, model)

		print(f"Training Loss:   {train_loss}")
		print(f"Validation Loss: {val_loss}")
		print(f"Best Val Loss:   {scheduler.best}")
		print(f"Learning Rate:   {optimizer.param_groups[0]['lr']}")
		print(f"LR Patience:     {scheduler.num_bad_epochs}/{scheduler.patience}")
		print(f"Early Stop:      {early_stopping.counter}/{early_stopping.patience}\n")

		# Update learning rate
		args.init_lr = optimizer.param_groups[0]['lr']

		# Check for early stopping
		if early_stopping.early_stop:
			print(f'Stopping Training. Validation loss has not improved in {args.early_stop_patience} epochs')
			break

		# Save checkpoint parameters
		if (epoch+1) % args.checkpoint_freq == 0:
			checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pt')
			torch.save({'model': model.state_dict(), 'args': args}, checkpoint_path)
			print(f'Checkpoint saved to: {checkpoint_path}\n')

	print('\nTraining complete! Model parameters saved to:')
	print(trained_model_path)


def main():
	args = options()
	print('')

	# Set training device
	device = get_device(args.device)

	# Initialize model
	model = load_model(CSGModel.num_shapes, CSGModel.num_operations, args, device)
	loss_func = Loss(PRIM_LOSS_WEIGHT, SHAPE_LOSS_WEIGHT, OP_LOSS_WEIGHT).to(device)
	optimizer = AdamW(model.parameters(), lr=args.init_lr)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience, threshold=args.lr_threshold, threshold_mode='rel')
	scaler = torch.amp.GradScaler(enabled=not args.disable_amp)

	# Load training set
	(args.output_dir, args.checkpoint_dir) = create_out_dir(args)
	(train_split, val_split, _) = load_data_splits(args, DATA_SPLIT, device)
	train_dataset = PointDataset(train_split, device, args, "Loading Training Set")
	val_dataset = PointDataset(val_split, device, args, "Loading Validation Set")

	train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=not args.keep_last_batch)
	val_sampler = BatchSampler(RandomSampler(val_dataset), batch_size=args.batch_size, drop_last=not args.keep_last_batch)

	train_loader = DataLoader(train_dataset, sampler=train_sampler)
	val_loader = DataLoader(val_dataset, sampler=val_sampler)

	# Save settings to file
	settings_path = os.path.join(args.output_dir, 'settings.yml')

	with open(settings_path, 'w') as out_path:
		yaml.dump(args.__dict__, out_path, sort_keys=False)

	# Train model
	print('')
	train(model, loss_func, optimizer, scheduler, scaler, train_loader, val_loader, args, device)


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
		print('\nProgram interrupted by keyboard input')
	finally:
		print('\nClearing GPU cache and quitting')
		torch.cuda.empty_cache()
