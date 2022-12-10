import os
import glob
import numpy as np


# Create output directory for trained model and temporary files
def create_out_dir(args):
	# Use parent directory name as dataset name
	dataset_name = os.path.basename(os.path.normpath(args.data_dir))
	# Create output folder name from settings
	output_folder = dataset_name + '_' + str(args.sample_dist) + 'dist_' + str(args.num_input_points) +\
	'input_points_' + str(args.num_loss_points) + 'loss_points_' + str(args.num_prims) + 'prims'
	output_dir = os.path.join(args.output_dir, output_folder)

	# Create output directory
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	elif len(os.listdir(output_dir)) != 0 and not args.overwrite:
		raise Exception('The output folder "%s" is already populated' % output_dir)

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
		raise Exception('No .npy data files found in directory "%s"' % data_dir)

	return file_rel_paths


# Write each item of a list to a new line in a file
def save_list(file_path, list):
	with open(file_path, 'w') as f:
		for item in list:
			f.write(f"{item}\n")


# Find and save all near-surface point samples
def uniform_to_surface_data(data_dir, uniform_rel_paths, output_dir, sample_dist):
	surface_points_dir = os.path.join(output_dir, 'near_surface_samples')

	for uniform_rel_path in uniform_rel_paths:
		# Select points for which the distance to the surface is within the threshold
		uniform_path = os.path.join(data_dir, uniform_rel_path)
		uniform_points = np.load(uniform_path)
		surface_points_rows = np.where(abs(uniform_points[:,3]) <= sample_dist)
		surface_points = uniform_points[surface_points_rows]

		# Save selected near-surface points
		surface_path = os.path.join(surface_points_dir, uniform_rel_path)
		os.makedirs(os.path.dirname(surface_path), exist_ok=True)
		np.save(surface_path, surface_points)

	return surface_points_dir