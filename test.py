
import argparse
import os
from pathlib import Path

import torch

from reconstruct import load_model
from utilities.data_processing import BEST_MODEL_FILE, find_file_paths, get_test_set
from utilities.device_utils import get_device


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--train_output_path', type=str, help='Path to training output directory. The best model and saved test set are automatically used.')
	parser.add_argument('--mesh_data_dir', type=str, help='Path to the root directory containing the mesh files referenced by the saved test set file. Only needed when train_output_path is provided.')
	parser.add_argument('--model_params', type=str, help='Path to a trained model pytorch file. Only needed when train_output_path is not provided.')
	parser.add_argument('--test_set_path', type=str, help='Path to a directory containing mesh files to test on. Overwrites the saved test set when train_output_path is also provided.')
	parser.add_argument('--num_cascades', type=int, help='Number of cascades to output before running tests. Defaults to the maximum number of cascades used during training.')
	parser.add_argument('--num_acc_points', type=int, default=30000, help='Number of points to use when computing accuracy.')
	parser.add_argument('--recon_resolution', type=int, default=256, help='Voxel resolution to use for the marching cubes algorithm when computing accuracy.')
	parser.add_argument('--device', type=str, default=[], nargs='*', help='Select preferred inference device. Evaluation only supports a single device')

	args = parser.parse_args()

	if not args.train_output_path:
		if not args.model_params:
			print('When train_output_path is not set, model_params must be set to a valid trained model file.')
			exit()

		if not args.test_set_path:
			print('When train_output_path is not set, test_set_path must be set to a directory containing test mesh files.')
			exit()

		if args.mesh_data_dir:
			print('mesh_data_dir should only be set when train_output_path is also set.')
			exit()
	else:
		if not args.mesh_data_dir and not args.test_set_path:
			print('When train_output_path is used, either mesh_data_dir or test_set_path must be set.')
			exit()

	return args


def test(model, test_samples):
	print(len(test_samples))
	return


def main():
	args = options()
	print('')

	# Assert a single inference device.
	if len(args.device) > 1:
		print('Evaluation only supports one device for inference. Select one device or select "None" to automatically select one.')
		exit()

	if len(args.device) > 0:
		args.device = args.device[0]

	device = get_device(args.device, cpu_allowed=True)

	# Load test files specified in the model output settings.
	if args.train_output_path:
		test_set_names = get_test_set(args.train_output_path)
		test_set_paths = find_file_paths(args.mesh_data_dir, test_set_names)
		args.model_params = os.path.join(args.train_output_path, BEST_MODEL_FILE)
	# Load all mesh files in test set directory and save to a list.
	else:
		target_dir = Path(args.test_set_path)
		test_set_paths = [str(path.resolve()) for path in target_dir.rglob('*') if path.is_file()]

	# Load model from file.
	(model, saved_args, init_model_state_dict, prev_cascades_list) = load_model(args.model_params, device, args.num_cascades)

	test(model, test_set_paths)



if __name__ == '__main__':
	main()
	