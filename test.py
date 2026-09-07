
import argparse
import os
from pathlib import Path
import signal
import sys
import traceback

import torch
from tqdm import tqdm
from wakepy import keep
from reconstruct import compute_chamfer_distance_mesh, compute_recon_loss, load_mesh_and_samples, load_model, model_inference
from utilities.csg_to_mesh import csg_to_mesh
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
	parser.add_argument('--recon_resolution', type=int, default=512, help='Voxel resolution to use for the marching cubes algorithm when computing accuracy.')
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


def test(model_params, num_acc_points, num_cascades, recon_resolution, device, test_samples):
	# Load model from file.
	(model, saved_args, init_model_state_dict, prev_cascades_list) = load_model(model_params, device, num_cascades)

	recon_loss = 0
	chamfer_dist = 0

	for sample_file in tqdm(test_samples):
		# For model inference.
		target_mesh, uniform_samples, near_surface_samples, surface_points = load_mesh_and_samples(sample_file, saved_args, num_acc_points, device)
		csg_model = model_inference(model, saved_args, init_model_state_dict, prev_cascades_list, near_surface_samples, uniform_samples)
		recon_mesh = csg_to_mesh(csg_model, recon_resolution)[0]

		# Compute accuracy metrics.
		recon_loss += compute_recon_loss(near_surface_samples, uniform_samples, surface_points, csg_model, saved_args.loss_metric, saved_args.excess_loss_weight)
		chamfer_dist += compute_chamfer_distance_mesh(target_mesh, recon_mesh, num_acc_points, device)

	mean_recon_loss = recon_loss / len(test_samples)
	mean_chamfer_dist = chamfer_dist / len(test_samples)

	return (mean_recon_loss, mean_chamfer_dist)


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

	# Test model.
	mean_recon_loss, mean_chamfer_dist = test(args.model_params, args.num_acc_points, args.num_cascades, args.recon_resolution, device, test_set_paths)

	print(f'Reconstruction Loss: {mean_recon_loss}')
	print(f'Chamfer Distance: {mean_chamfer_dist}')


if __name__ == '__main__':
	def exit_handler():
		print('\nClearing GPU cache and quitting')
		torch.cuda.empty_cache()
		sys.exit()

	# Catch CTRL+Z force shutdown
	signal.signal(signal.SIGTSTP, lambda _signum, _frame: exit_handler())

	try:
		with keep.running():
			main()
	# Catch CTRL+C force shutdown
	except KeyboardInterrupt:
		print('\nProgram interrupted by keyboard input')
	except Exception:
		print(traceback.format_exc())
	finally:
		exit_handler()
