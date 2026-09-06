
import argparse

import torch

from utilities.data_augmentation import RotationAxis
from utilities.data_processing import get_test_set_absolute_paths, load_list


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--train_output_path', type=str, help='Path to training output directory containing the model, settings, and test set files.')
	parser.add_argument('--model_params', type=str, help='Path to a trained model pytorch file. Only needed when train_output_path is not provided.')
	parser.add_argument('--test_set_path', type=str, help='Path to a directory containing sample files to test on. Overwrites the saved test set when train_output_path is also provided.')
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
			print('When train_output_path is not set, test_set_path must be set to a directory of sample file.')
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

	# Load test files specified in the model output settings.
	if args.train_output_path:
		test_set_paths = get_test_set_absolute_paths(args.train_output_path)
	# Load all test files in directory to a list.
	else:
		test_set_paths = load_list(args.test_set_path)

	test(None, test_set_paths)



if __name__ == '__main__':
	main()
	