import argparse
import os
import signal
import sys
import torch
import traceback

from torch.utils.data import Subset

from losses.loss import Loss
from losses.reconstruction_loss import ReconstructionLoss
from utilities.data_processing import create_out_dir, read_dataset_settings, save_dataset_settings, LATEST_MODEL_FILE
from utilities.data_augmentation import get_augment_parser, RotationAxis
from utilities.train_utils import load_data_splits, load_model, train, init_training_params
from utilities.training_logger import TrainingLogger


# Percentage of data to use for training, validation, and testing
DATA_SPLIT = [0.8, 0.1, 0.1]


# Parse commandline arguments
def options():
	# Parse and handle Help argument
	help_parser = get_help_parser()
	help_arg, remaining_args = help_parser.parse_known_args()

	if help_arg.help or not remaining_args:
		print_help()
		exit()

	# Parse data settings
	data_parser = get_data_parser()
	args, remaining_args = data_parser.parse_known_args(args=remaining_args)

	# Enforce prerequisites
	if args.resume_training and not args.model_path:
		print('Cannot use the resume_training option without providing the --model_path option')
		exit()

	if args.model_path and getattr(args, 'continue'):
		print('Cannot use both the --model_path and --continue options simultaneously')
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
		args, _unused_args = augment_parser.parse_known_args(args=remaining_args, namespace=args)

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

	# Retrieve loss metric
	args.loss_metric = args.loss_metric[0] if len(args.loss_metric) > 0 else None

	# Retrieve loss sampling method
	args.loss_sampling_method = args.loss_sampling_method[0] if len(args.loss_sampling_method) > 0 else None

	# Configure subtract operation
	if args.disable_sub_operation or args.schedule_sub_weight:
		args.sub_weight = 0
	else:
		args.sub_weight = 1

	# Configure cascade scheduling
	if args.cascade_schedule_epochs <= 0:
		args.no_schedule_cascades = True

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

	data_group.add_argument('--data_dir', type=str, help='Parent directory of the SDF Dataset (data in subdirectories is included). Required unless the --model_path and --resume_training options are provided')
	data_group.add_argument('--sub_dir', type=str, help='A subdirectory of of the parent SDF dataset to train on. The subdirectory must be present in the nea-surface, surface, and uniform directories.')
	data_group.add_argument('--output_dir', type=str, default='./output', help='Output directory for checkpoints, trained model, and augmented dataset')
	data_group.add_argument('--model_path', type=str, default='', help='Load parameters and settings from saved model file. Provided arguments overwrite all the saved arguments except for network model settings')
	data_group.add_argument('--resume_training', default=False, action='store_true', help='If a model path is supplied, resume training of the model with the original training data')
	data_group.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')
	data_group.add_argument('--continue', default=False, action='store_true', help='Resume training if the output directory exists. model_path is inferred from arguments and resume_training and overwrite are set to true.')

	return data_parser


def get_model_parser():
	model_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	model_group = model_parser.add_argument_group('MODEL SETTINGS')

	# Model settings
	model_group.add_argument('--num_input_points', type=int, default=1024, help='Number of points to use from each input sample (Memory requirement scales linearly with num_input_points)')
	model_group.add_argument('--num_loss_points', type=int, default=2048, help='Number of points to use when computing the loss')
	model_group.add_argument('--num_val_acc_points', type=int, default=1024, help='Number of points to use when computing validation metrics')
	model_group.add_argument('--val_sample_dist', type=float, default=0.01, help='Maximum distance tolerance of approximate surface samples when computing validation metrics.')
	model_group.add_argument('--num_prims', type=int, default=3, help='Number of primitives to generate before backpropagating (Memory requirement scales with num_prims)')
	model_group.add_argument('--num_cascades', type=int, default=1, help='Number of refinement passes before back-propagating (Total generated primitives = num_prims * num_cascades)')
	model_group.add_argument('--no_extended_input', default=False, action='store_true', help='Exclude the additional volume parameters from the model input.')
	model_group.add_argument('--no_blending', default=False, action='store_true', help='Disable primitive blending')
	model_group.add_argument('--no_roundness', default=False, action='store_true', help='Disable primitive rounding')
	model_group.add_argument('--no_batch_norm', default=False, action='store_true', help='Disable batch normalization')
	model_group.add_argument('--loss_sampling_method', type=str.upper, default=[Loss.UNIFIED_SAMPLING], choices=Loss.loss_sampling_methods, nargs=1, help="TARGET_SAMPLING samples loss points from only the target shape and UNIFIED_SAMPLING samples from both target and reconstruction shapes.")
	model_group.add_argument('--surface_uniform_ratio', type=float, default=0.5, help='Percentage of near-surface samples to select. 0 for only uniform samples and 1 for only near-surface samples')
	model_group.add_argument('--decoder_layers', nargs='+', type=int, default=[], help='List of hidden layers to add to the decoder network')
	model_group.add_argument('--back_prop_recon_input', default=False, action='store_true', help='Backpropagate through the reconstruction input sample and all previous refinement iterations.')

	return model_parser


def get_training_parser(suppress_default=False):
	argument_default = argparse.SUPPRESS if suppress_default else None
	training_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS, argument_default=argument_default)
	training_group = training_parser.add_argument_group('TRAINING SETTINGS')

	# Training settings
	training_group.add_argument('--batch_size', type=int, default=32, help='Mini-batch size. When set to 1, batch normalization is disabled')
	training_group.add_argument('--keep_last_batch', default=False, action='store_true', help='Train on remaining data samples at the end of each epoch')
	training_group.add_argument('--max_epochs', type=int, default=2000, help='Maximum number of epochs to train')
	training_group.add_argument('--loss_metric', type=str.upper, default=[ReconstructionLoss.L1_LOSS_FUNC], choices=ReconstructionLoss.loss_metrics, nargs=1, help='Reconstruction loss metric to use when training')
	training_group.add_argument('--clamp_dist', type=float, default=0.1, help='Restrict the loss computation to a maximum specified distance from the target shape')
	training_group.add_argument('--backprop_cascades', default=False, action='store_true', help='When enabled, backpropagate through each cascade separately. When disabled, backpropagate through all cascades. Disabling the setting requires more GPU memory but gives the model more context.')
	training_group.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate')
	training_group.add_argument('--lr_factor', type=float, default=0.1, help='Learning rate reduction factor')
	training_group.add_argument('--lr_patience', type=int, default=20, help='Number of training epochs without improvement before the learning rate is adjusted')
	training_group.add_argument('--lr_threshold', type=float, default=0.01, help='Minimum recognized percentage of improvement over previous loss')
	training_group.add_argument('--early_stop_patience', type=int, default=40, help='Number of training epochs without improvement before training terminates')
	training_group.add_argument('--early_stop_threshold', type=float, default=0.01, help='Minimum recognized percentage of improvement over previous loss')
	training_group.add_argument('--disable_sub_operation', default=False, action='store_true', help='Disable the subtract operation by setting the weight to 0')
	training_group.add_argument('--schedule_sub_weight', default=False, action='store_true', help='Start the subtract operation weight at 0 and gradually increase it to 1')
	training_group.add_argument('--sub_schedule_start_epoch', type=int, default=10, help='Epoch to start the subtract operation weight scheduler')
	training_group.add_argument('--sub_schedule_end_epoch', type=int, default=30, help='Epoch to complete the subtract operation weight scheduler')
	training_group.add_argument('--no_schedule_cascades', default=False, action='store_true', help='Begin training with all refinement iterations enabled rather than progressively added with a scheduler.')
	training_group.add_argument('--cascade_schedule_epochs', type=int, default=10, help='Number of epochs to train before adding a new refinement iteration.')
	training_group.add_argument('--checkpoint_freq', type=int, default=10, help='Number of epochs to train for before saving model parameters')
	training_group.add_argument('--device', type=str, default='', help='Select preferred training device')
	training_group.add_argument('--disable_amp', default=False, action='store_true', help='Disable Automatic Mixed Precision')

	return training_parser


def get_online_augment_parser(suppress_default=False):
	return get_augment_parser('ONLINE AUGMENT SETTINGS', suppress_default)


# Determine device to train on
def get_device(device=None):
	if device == 'cpu':
		raise Exception('Only CUDA devices are supported.')
	elif device:
		return torch.device(device)
	elif torch.cuda.is_available():
		return torch.device('cuda')
	else:
		raise Exception('No CUDA devices are available.')


# Set model_path, resume_training, and overwrite options if continue option is provided
def process_continue(args):
	if getattr(args, 'continue'):
		# Get output directory from argumnets
		(args.output_dir, args.checkpoint_dir) = create_out_dir(args)

		# Check that the directory exists
		if not os.path.exists(args.output_dir):
			print(f'Output directory does not exist in expected location: {args.output_dir}')
			exit()

		latest_model_path = os.path.join(args.output_dir, LATEST_MODEL_FILE)

		# Check that the model file exists
		if not os.path.isfile(latest_model_path):
			print(f'Model parameter file does not exist in expected location: {latest_model_path}')
			exit()

		args.model_path = latest_model_path
		args.resume_training = True
		args.overwrite = True


# Load saved settings if a model path is provided
def load_saved_settings(args):
	if args.model_path:
		torch.serialization.add_safe_globals([argparse.Namespace, Subset, RotationAxis])
		saved_settings_dict = torch.load(args.model_path, weights_only=True)
		model_params = saved_settings_dict['model']
		return (saved_settings_dict, model_params)
	else:
		return (None, None)


# Initialize output directories and training split
def init_output(args, device, saved_settings_dict=None):
	# Load settings from file if resuming training. Otherwise, initialize output directories and training split
	if args.resume_training:
		args.data_dir = saved_settings_dict['data_dir']
		args.output_dir = saved_settings_dict['output_dir']
		data_splits = saved_settings_dict['data_splits']
		training_results = saved_settings_dict['training_results']
		training_logger = TrainingLogger(args.output_dir, 'training_results', args.loss_metric, training_results)
	else:
		(args.output_dir, args.checkpoint_dir) = create_out_dir(args)
		data_splits = load_data_splits(args, DATA_SPLIT, device)
		training_logger = TrainingLogger(args.output_dir, 'training_results', args.loss_metric)

	return (data_splits, training_logger)


def main():
	args = options()
	print('')

	# Set training device
	device = get_device(args.device)

	# Initialize options and output
	process_continue(args)
	saved_settings_dict, model_params = load_saved_settings(args)
	data_splits, training_logger = init_output(args, device, saved_settings_dict)

	# Read settings from dataset
	dataset_settings = read_dataset_settings(args.data_dir)
	args.sample_dist = dataset_settings['sample_dist']

	# Save settings to file
	save_dataset_settings(args.output_dir, args.__dict__)
	print('')

	# Train model
	training_params = init_training_params(training_logger, data_splits, args, device, model_params)
	train(*training_params, training_logger, data_splits, args, device)


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
