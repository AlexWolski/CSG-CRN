import os
import glob
import numpy as np
from tqdm import tqdm


# Subdirectory names
UNIFORM_FOLDER = 'uniform'
SURFACE_FOLDER = 'surface'
NEAR_SURFACE_FOLDER = 'near-surface'

# Dataset metadata filenames
SETTINGS_FILE = 'settings.yml'
SAMPLE_LIST_FILE = 'files.txt'


# Create output directory for trained model and temporary files
def create_out_dir(args):
	# Use parent directory name as dataset name
	dataset_name = os.path.basename(os.path.normpath(args.data_dir))

	# Create output folder name from settings
	output_folder = dataset_name + '_' + str(args.sample_dist) + 'dist_' + str(args.num_input_points) +\
	'input_points_' + str(args.num_loss_points) + 'loss_points_' + str(args.num_prims) + 'prims'

	if args.no_blending:
		output_folder += '_no_blending'

	if args.no_roundness:
		output_folder += '_no_roundness'

	output_dir = os.path.join(args.output_dir, output_folder)

	# Create output directory
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	elif len(os.listdir(output_dir)) != 0 and not args.overwrite:
		err_msg = f'The output folder "{output_dir}" is already populated. Choose another directory, use the --overwrite option, or use the --resume option.'
		raise Exception(err_msg)

	# Create checkpoint folder
	checkpoint_dir = get_checkpoint_dir(output_dir)
	os.makedirs(checkpoint_dir, exist_ok=True)

	return (output_dir, checkpoint_dir)


# Return the checkpoints directory path
def get_checkpoint_dir(output_dir):
	return os.path.join(output_dir, 'checkpoints')


# Read file list from a dataset directory
def get_data_files(data_dir):
	sample_list_path = os.path.join(data_dir, SAMPLE_LIST_FILE)

	if os.path.isfile(sample_list_path):
		return load_list(sample_list_path)
	else:
		raise FileNotFoundError(f'Unable to find dataset settings file: {sample_list_path}')


# Write each item of a list to a new line in a file
def save_list(file_path, output_list):
	with open(file_path, 'w') as f:
		f.write('\n'.join(output_list))


# Read all the lines of a file to a list
def load_list(file_path):
	list_data = []

	with open(file_path, 'r') as f:
		for item in f:
			list_data.append(item.rstrip('\n'))

	return list_data
