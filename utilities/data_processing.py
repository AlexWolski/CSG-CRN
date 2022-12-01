import os
import glob
import numpy as np


# Create output directory for trained model and temporary files
def create_out_dir(args):
	# Use parent directory name as dataset name
	dataset_name = os.path.basename(os.path.normpath(args.data_dir))
	# Create output folder name from settings
	output_folder = dataset_name + '_' + str(args.sample_dist) + 'dist_' + str(args.num_points) + 'points_' + str(args.num_prims) + 'prims'
	output_path = os.path.join(args.output_dir, output_folder)

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	elif len(os.listdir(output_path)) != 0:
		raise Exception('The output folder "%s" is already populated' % output_path)

	return output_path


# Find all data files in a given directory
def get_data_files(data_dir):
	# Recursively find all npy files in parent directory
	file_paths = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)

	if len(file_paths) == 0:
		raise Exception('No .npy data files found in directory "%s"' % data_dir)
	
	return file_paths


# Write each item of a list to a new line in a file
def save_list(file_path, list):
	with open(file_path, 'w') as f:
		for item in list:
			f.write(f"{item}\n")


# Find and save all near-surface point samples
def uniform_to_surface_data(uniform_files, sample_dist, output_path):
	filenames = []

	surface_points_dir = os.path.join(output_path, 'near_surface_samples')
	os.mkdir(surface_points_dir)

	for uniform_file in uniform_files:
		# Select points for which the distance to the surface is within the threshold
		uniform_points = np.load(uniform_file)
		surface_points_rows = np.where(abs(uniform_points[:,3]) <= sample_dist)
		surface_points = uniform_points[surface_points_rows]

		# Save selected near-surface points
		filename = os.path.basename(uniform_file)
		filenames.append(filename)
		surface_points_path = os.path.join(surface_points_dir, filename)
		np.save(surface_points_path, surface_points)

	return (surface_points_dir, filenames)