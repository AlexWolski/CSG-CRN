import os
import glob
import numpy as np
from tqdm import tqdm


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
		err_msg = f'The output folder "{output_dir}" is already populated. Use another directory or the --overwrite option.'
		raise Exception(err_msg)

	# Create checkpoint folder
	checkpoint_dir = os.path.join(output_dir, 'checkpoints')
	os.makedirs(checkpoint_dir, exist_ok=True)

	return (output_dir, checkpoint_dir)


# Find all data files in a given directory
def get_data_files(data_dir):
	# Find all npy files in parent directory
	file_paths = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)
	file_rel_paths = [os.path.relpath(file_path, data_dir) for file_path in file_paths]

	if len(file_rel_paths) == 0:
		err_msg = f'No .npy data files found in directory "{data_dir}"'
		raise Exception(err_msg)

	return file_rel_paths


# Write each item of a list to a new line in a file
def save_list(file_path, list):
	with open(file_path, 'w') as f:
		for item in list:
			f.write(f"{item}\n")


# Select points within a threshold distance of the surface
def select_surface_points(samples, sample_dist):
	surface_sample_rows = np.where(abs(samples[:,3]) <= sample_dist)
	return samples[surface_sample_rows]


# Select near-surface point samples and consolidate sample length
def pre_process_sample(args, data_sample):
	total_points = args.num_input_points + args.num_loss_points

	# Select near-surface points
	if args.sample_method[0] == 'near-surface':
		data_sample = select_surface_points(data_sample, args.sample_dist)

	# Skip samples that don't contain enough points
	if data_sample.shape[0] < total_points:
		return None

	# Randomly select the required number of points
	if data_sample.shape[0] > total_points:
		select_rows = np.random.choice(data_sample.shape[0], total_points, replace=False)
		data_sample = data_sample[select_rows]

	return data_sample