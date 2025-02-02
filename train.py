import argparse
import os
import math
import signal
import sys
import torch
import traceback
import yaml

from tqdm import tqdm
from torch import autocast
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.distributions.uniform import Uniform
from torch.optim import AdamW, lr_scheduler

from losses.loss import Loss
from networks.csg_crn import CSG_CRN
from utilities.csg_model import CSGModel
from utilities.data_processing import *
from utilities.datasets import PointDataset
from utilities.data_augmentation import get_augment_parser, RotationAxis
from utilities.early_stopping import EarlyStopping
from utilities.training_logger import TrainingLogger


# Weights for regularization loss
PROXIMITY_LOSS_WEIGHT = 1
# Percentage of data to use for training, validation, and testing
DATA_SPLIT = [0.8, 0.1, 0.1]


# Parse commandline arguments
def options():
	# Parse and handle Help argument
	help_parser = get_help_parser()
	args, remaining_args = help_parser.parse_known_args()

	if args.help or not remaining_args:
		print_help()
		exit()

	# Parse data settings
	data_parser = get_data_parser()
	args, remaining_args = data_parser.parse_known_args(args=remaining_args, namespace=args)

	# Enforce prerequisites
	if args.resume_training and not args.model_path:
		print('Cannot use the resume_training option without providing the --model_path option')
		exit()

	if not args.data_dir and not (args.model_path or args.resume_training):
		print('Missing --data_dir option. Either provide --data_dir option or both --model_path and --resume_training')
		exit()

	# Load settings from saved model file
	if args.model_path:
		print('\nLoading Arguments From Model File:')
		print(os.path.abspath(args.model_path))

		# Cache data settings
		data_args = args

		# Load arguments from model file
		torch.serialization.add_safe_globals([argparse.Namespace, Subset, RotationAxis])
		args = torch.load(args.model_path, weights_only=True)['args']

		# Apply data settings
		for data_arg_name in vars(data_args):
			arg_value = getattr(data_args, data_arg_name)
			setattr(args, data_arg_name, arg_value)

		# Apply training settings
		training_parser = get_training_parser(suppress_default=True)
		augment_parser = get_online_augment_parser(suppress_default=True)
		args, remaining_args = training_parser.parse_known_args(args=remaining_args, namespace=args)
		augment_parser.parse_args(args=remaining_args, namespace=args)

	# Parse remaining arguments
	else:
		model_parser = get_model_parser()
		training_parser = get_training_parser()
		augment_parser = get_online_augment_parser()
		args, remaining_args = model_parser.parse_known_args(args=remaining_args, namespace=args)
		args, remaining_args = training_parser.parse_known_args(args=remaining_args, namespace=args)
		augment_parser.parse_args(args=remaining_args, namespace=args)


	# Expand paths
	args.data_dir = os.path.abspath(args.data_dir) if args.data_dir else None
	args.model_path = os.path.abspath(args.model_path) if args.model_path else None
	args.output_dir = os.path.abspath(args.output_dir)

	# Disable batch norm for SGD
	args.no_batch_norm = True if args.batch_size == 1 else args.no_batch_norm

	# Print arguments
	print('\nArguments:')
	print('----------')

	for arg_name in vars(args):
		print('{0:20} - {1}'.format(arg_name, getattr(args, arg_name)))

	return args


def print_help():
	parsers = [get_data_parser(), get_model_parser(), get_training_parser(), get_online_augment_parser()]

	for parser in parsers:
		print('\n')
		parser.print_help()


def get_help_parser():
	help_parser = argparse.ArgumentParser(add_help=False)
	help_parser.add_argument('-h', '--help', default=False, action='store_true', help='Print help text')

	return help_parser


def get_data_parser():
	data_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	data_group = data_parser.add_argument_group('DATA SETTINGS')

	data_group.add_argument('--data_dir', type=str, help='Dataset parent directory (data in subdirectories is included). Required unless the --model_path and --resume_training options are provided')
	data_group.add_argument('--output_dir', type=str, default='./output', help='Output directory for checkpoints, trained model, and augmented dataset')
	data_group.add_argument('--model_path', type=str, default='', help='Load parameters and settings from saved model file. Provided arguments overwrite all the saved arguments except for network model settings')
	data_group.add_argument('--resume_training', default=False, action='store_true', help='If a model path is supplied, resume training of the model with the original training data')
	data_group.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')
	data_group.add_argument('--skip_preprocess', default=False, action='store_true', help='Skip the pre-processing step if the provided data_dir already contains samples of the proper length and sampling method')

	return data_parser


def get_model_parser():
	model_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	model_group = model_parser.add_argument_group('MODEL SETTINGS')

	# Model settings
	model_group.add_argument('--num_input_points', type=int, default=1024, help='Number of points to use from each input sample (Memory requirement scales linearly with num_input_points)')
	model_group.add_argument('--num_loss_points', type=int, default=20000, help='Number of points to use when computing the loss')
	model_group.add_argument('--num_prims', type=int, default=3, help='Number of primitives to generate before computing loss (Memory requirement scales with num_prims)')
	model_group.add_argument('--num_iters', type=int, default=10, help='Number of refinement iterations to train for (Total generated primitives = num_prims x num_iters)')
	model_group.add_argument('--no_blending', default=False, action='store_true', help='Disable primitive blending')
	model_group.add_argument('--no_roundness', default=False, action='store_true', help='Disable primitive rounding')
	model_group.add_argument('--no_batch_norm', default=False, action='store_true', help='Disable batch normalization')
	model_group.add_argument('--surface_uniform_ratio', type=float, default=0.5, help='Percentage of near-surface samples to select. 0 for only uniform samples and 1 for only near-surface samples')
	model_group.add_argument('--sample_dist', type=float, default=0.1, help='Maximum distance to object surface for near-surface sampling (must be >0)')
	model_group.add_argument('--decoder_layers', nargs='+', type=int, default=[], help='List of hidden layers to add to the decoder network')

	return model_parser


def get_training_parser(suppress_default=False):
	argument_default = argparse.SUPPRESS if suppress_default else None
	training_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS, argument_default=argument_default)
	training_group = training_parser.add_argument_group('TRAINING SETTINGS')

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

	return training_parser


def get_online_augment_parser(suppress_default=False):
	return get_augment_parser('ONLINE AUGMENT SETTINGS', suppress_default)


# Determine device to train on
def get_device(device=None):
	if device:
		return torch.device(device)
	elif torch.cuda.is_available():
		return torch.device('cuda')
	else:
		return torch.device('cpu')


# Prepare data files and load training dataset
def load_data_splits(args, data_split, device):
	# Load sample files
	file_rel_paths = get_data_files(args.data_dir)
	print(f'Found {len(file_rel_paths)} data files')

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

	print(f'Training set:\t{len(train_split.indices)} samples')
	print(f'Validation set:\t{len(val_split.indices)} samples')
	print(f'Testing set:\t{len(test_split.indices)} samples\n')

	return (train_split, val_split, test_split)


# Load CSG-CRN network model
def load_model(num_prims, num_shapes, num_operations, device, args, model_params=None):
	predict_blending = not args.no_blending
	predict_roundness = not args.no_roundness

	# Initialize model
	model = CSG_CRN(num_prims, num_shapes, num_operations, args.decoder_layers, predict_blending, predict_roundness, args.no_batch_norm).to(device)

	# Load model parameters if available
	if model_params:
		model.load_state_dict(model_params)

	return model


# Run a forwards pass of the network model
def model_forward(model, loss_func, target_input_samples, target_loss_samples, recon_input_samples, recon_loss_samples, args, device):
	# Load data
	(batch_size, num_input_points, _) = target_input_samples.size()

	# Initialize SDF CSG model
	csg_model = CSGModel(device)

	# Generate a set of primitives to add to the CSG model
	with autocast(device_type=device.type, dtype=torch.float16, enabled=not args.disable_amp):
		output_list = model(target_input_samples, recon_input_samples)

	# Add primitive to CSG model
	for output in output_list:
		csg_model.add_command(*output)

	# Compute loss
	loss = loss_func(target_loss_samples, recon_loss_samples, csg_model)

	return loss


# Iteratively predict primitives and propagate average loss
def train_one_epoch(model, loss_func, optimizer, scaler, train_loader, args, device, desc=''):
	total_train_loss = 0

	for (target_input_samples, target_loss_samples, recon_input_samples, recon_loss_samples) in tqdm(train_loader, desc=desc):
		# Forward pass
		batch_loss = model_forward(model, loss_func, target_input_samples, target_loss_samples, recon_input_samples, recon_loss_samples, args, device)
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
		for (target_input_samples, target_loss_samples, recon_input_samples, recon_loss_samples) in val_loader:
			batch_loss = model_forward(model, loss_func, target_input_samples, target_loss_samples, recon_input_samples, recon_loss_samples, args, device)
			total_val_loss += batch_loss.item()

	total_val_loss /= val_loader.__len__()
	return total_val_loss


# Save the model and settings to file
def save_model(model, args, data_splits, training_results, model_path):
	torch.save({
		'model': model.state_dict(),
		'args': args,
		'data_dir': args.data_dir,
		'output_dir': args.output_dir,
		'data_splits': data_splits,
		'training_results': training_results
	}, model_path)


# Train model for max_epochs or until stopped early
def train(model, loss_func, optimizer, scheduler, scaler, train_loader, val_loader, data_splits, args, device, training_logger):
	model.train(True)

	# Initialize early stopper
	trained_model_path = os.path.join(args.output_dir, 'best_model.pt')
	checkpoint_dir = get_checkpoint_dir(args.output_dir)
	save_best_model = lambda: save_model(model, args, data_splits, training_logger.get_results(), trained_model_path)
	early_stopping = EarlyStopping(args.early_stop_patience, args.early_stop_threshold, save_best_model)

	# Train until model stops improving or a maximum number of epochs is reached
	init_epoch = training_logger.get_last_epoch()+1 if training_logger.get_last_epoch() else 1

	for epoch in range(init_epoch, args.max_epochs+1):
		# Train model
		desc = f'Epoch {epoch}/{args.max_epochs}'
		train_loss = train_one_epoch(model, loss_func, optimizer, scaler, train_loader, args, device, desc)
		val_loss = validate(model, loss_func, val_loader, args, device)
		learning_rate = optimizer.param_groups[0]['lr']
		scheduler.step(val_loss)

		training_logger.add_result(epoch, train_loss, val_loss, learning_rate)
		early_stopping(val_loss)

		# Print and save epoch training results
		print(f"Training Loss:   {train_loss}")
		print(f"Validation Loss: {val_loss}")
		print(f"Best Val Loss:   {scheduler.best}")
		print(f"Learning Rate:   {learning_rate}")
		print(f"LR Patience:     {scheduler.num_bad_epochs}/{scheduler.patience}")
		print(f"Early Stop:      {early_stopping.counter}/{early_stopping.patience}\n")

		# Check for early stopping
		if early_stopping.early_stop:
			print(f'Stopping Training. Validation loss has not improved in {args.early_stop_patience} epochs')
			break

		# Save checkpoint parameters
		if epoch % args.checkpoint_freq == 0:
			checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
			save_model(model, args, data_splits, training_logger.get_results(), checkpoint_path)
			print(f'Checkpoint saved to: {checkpoint_path}\n')

	print('\nTraining complete! Model parameters saved to:')
	print(trained_model_path)


def main():
	args = options()
	print('')

	# Set training device
	device = get_device(args.device)

	# Load saved settings if a model path is provided
	if args.model_path:
		torch.serialization.add_safe_globals([argparse.Namespace, Subset, RotationAxis])
		saved_settings_dict = torch.load(args.model_path, weights_only=True)
		model_params = saved_settings_dict['model']

	# Load settings from file if resuming training. Otherwise, initialize output directories and training split
	if args.resume_training:
		args.data_dir = saved_settings_dict['data_dir']
		args.output_dir = saved_settings_dict['output_dir']
		data_splits = saved_settings_dict['data_splits']
		training_results = saved_settings_dict['training_results']
		training_logger = TrainingLogger(args.output_dir, 'training_results', training_results)
	else:
		(args.output_dir, checkpoint_dir) = create_out_dir(args)
		data_splits = load_data_splits(args, DATA_SPLIT, device)
		training_logger = TrainingLogger(args.output_dir, 'training_results')

	# Initialize model
	model = load_model(args.num_prims, CSGModel.num_shapes, CSGModel.num_operations, device, args, model_params if args.resume_training else None)
	loss_func = Loss(PROXIMITY_LOSS_WEIGHT).to(device)
	current_lr = training_logger.get_last_lr() if training_logger.get_last_lr() else args.init_lr
	optimizer = AdamW(model.parameters(), lr=current_lr)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience, threshold=args.lr_threshold, threshold_mode='rel')
	scaler = torch.amp.GradScaler(enabled=not args.disable_amp)

	# Load training set
	(train_split, val_split, test_split) = data_splits
	checkpoint_dir = get_checkpoint_dir(args.output_dir)

	if not (train_dataset := PointDataset(train_split, device, args, "Training Set")):
		return

	if not (val_dataset := PointDataset(val_split, device, args, "Validation Set")):
		return

	train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=not args.keep_last_batch)
	val_sampler = BatchSampler(RandomSampler(val_dataset), batch_size=args.batch_size, drop_last=not args.keep_last_batch)

	# The PointDataset class has a custom __getitem__ function so the collate function is unneeded
	collate_fn = lambda data: data[0]
	train_loader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fn)
	val_loader = DataLoader(val_dataset, sampler=val_sampler, collate_fn=collate_fn)

	# Save settings to file
	settings_path = os.path.join(args.output_dir, 'settings.yml')

	with open(settings_path, 'w') as out_path:
		yaml.dump(args.__dict__, out_path, sort_keys=False)

	# Train model
	print('')
	train(model, loss_func, optimizer, scheduler, scaler, train_loader, val_loader, data_splits, args, device, training_logger)


if __name__ == '__main__':
	def exit_handler():
		print('\nClearing GPU cache and quitting')
		torch.cuda.empty_cache()
		sys.exit()

	# Catch CTRL+Z force shutdown
	signal.signal(signal.SIGTSTP, lambda _signum, _frame: exit_handler())

	try:
		main()
	# Catch CTRL+C force shutdown
	except KeyboardInterrupt:
		print('\nProgram interrupted by keyboard input')
	except Exception:
		print(traceback.format_exc())
	finally:
		exit_handler()
