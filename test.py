
import argparse
from datetime import datetime, timedelta
import io
import math
import os
import signal
import sys
import traceback

import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset
from tqdm import tqdm
from wakepy import keep
from losses.reconstruction_loss import ReconstructionLoss
from reconstruct import load_mesh_and_samples, load_model, model_inference
from utilities.accuracy_metrics import EMD, compute_chamfer_distance
from utilities.csg_to_mesh import csg_to_mesh
from utilities.data_augmentation import RotationAxis
from utilities.data_processing import BEST_MODEL_FILE, TEST_RESULTS_FILE, find_file_paths, get_test_set
from utilities.device_utils import get_devices
from utilities.file_utils import get_mesh_files
from utilities.sampler_utils import sample_points_mesh_surface


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--train_output_path', type=str, help='Path to training output directory. The best model and test set in the directory are used. Test results are stored in dirctory.')
	parser.add_argument('--mesh_data_dir', type=str, help='Path to the root directory containing the mesh files referenced by the saved test set file.')
	parser.add_argument('--model_params', type=str, help='Path to a trained model pytorch file. Only needed when train_output_path is not provided.')
	parser.add_argument('--test_set_path', type=str, help='Path to a directory containing mesh files to test on. Overwrites the saved test set when train_output_path is also provided.')
	parser.add_argument('--result_file_path', type=str, help='Optional file path to store the test results in. When provided, results will not be stored in train_output_path.')

	parser.add_argument('--num_cascades', type=int, help='Number of cascades to output before running tests. Defaults to the maximum number of cascades used during training.')
	parser.add_argument('--num_acc_points', type=int, default=10000, help='Number of points to use when computing accuracy.')
	parser.add_argument('--recon_resolution', type=int, default=512, help='Voxel resolution to use for the marching cubes algorithm when computing accuracy.')
	parser.add_argument('--device', type=str.lower, default=[], nargs='*', help='Select one or more CUDA devices. Select "all" to use all available cuda devices.')

	args = parser.parse_args()

	# Expand paths
	args.train_output_path = os.path.abspath(args.train_output_path) if args.train_output_path else None
	args.mesh_data_dir = os.path.abspath(args.mesh_data_dir) if args.mesh_data_dir else None
	args.model_params = os.path.abspath(args.model_params) if args.model_params else None
	args.test_set_path = os.path.abspath(args.test_set_path) if args.test_set_path else None
	args.result_file_path = os.path.abspath(args.result_file_path) if args.result_file_path else None

	if not args.train_output_path and not args.model_params:
		print('Either train_output_path or model_params must be set.')
		exit()

	if not args.mesh_data_dir and not args.test_set_path:
		print('Either mesh_data_dir or test_set_path must be set.')
		exit()

	return args


def test_worker(worker_index, devices, model_params, num_acc_points, num_cascades, recon_resolution, sample_splits, result_queue=None):
	device = devices[worker_index]
	test_samples = sample_splits[worker_index]

	# Set worker device.
	torch.cuda.set_device(device)

	# Each worker loads a separate copy of the model.
	(model, saved_args, init_model_state_dict, prev_cascades_list) = load_model(model_params, device, num_cascades)
	model.eval()

	recon_loss = ReconstructionLoss(saved_args.loss_metric, saved_args.excess_loss_weight)

	summed_recon_loss = 0.0
	summed_chamfer_dist = 0.0
	summed_earth_dist = 0.0
	skipped_samples = 0

	with torch.no_grad():
		for sample_file in tqdm(test_samples, desc=str(device), position=worker_index):
			# Reconstruct sample and generate a mesh.
			target_mesh, uniform_samples, near_surface_samples, surface_points = load_mesh_and_samples(sample_file, saved_args, num_acc_points, device)
			csg_model = model_inference(model, saved_args, init_model_state_dict, prev_cascades_list, near_surface_samples, uniform_samples)
			recon_mesh_list = csg_to_mesh(csg_model, recon_resolution)

			# Skip bad meshes.
			if not recon_mesh_list:
				skipped_samples += 1
				continue

			# Compute reconstruction loss with original training settings.
			summed_recon_loss += recon_loss.forward(near_surface_samples, uniform_samples, surface_points, csg_model).item()

			# Sample points from output mesh.
			target_points = sample_points_mesh_surface(target_mesh, num_acc_points).unsqueeze(0).to(device)
			recon_points = sample_points_mesh_surface(recon_mesh_list[0], num_acc_points).unsqueeze(0).to(device)

			# Compute accuracy metrics.
			summed_chamfer_dist += compute_chamfer_distance(target_points, recon_points, no_grad=True)
			summed_earth_dist += EMD(target_points[0], recon_points[0]).item()

	# All summed values are Python floats (moved off the GPU) so they can be returned to the parent process.
	result_queue.put((summed_recon_loss, summed_chamfer_dist, summed_earth_dist, skipped_samples))


def test(model_params, num_acc_points, num_cascades, recon_resolution, devices, test_samples):
	num_samples = len(test_samples)
	samples_per_gpu = math.ceil(len(test_samples) / len(devices))

	# Compute number of samples to process on each GPU.
	sample_splits = []

	for curr_sample in range(0, len(test_samples), samples_per_gpu):
		sample_splits.append(test_samples[curr_sample:curr_sample+samples_per_gpu])

	# Drop unused devices.
	num_workers = len(sample_splits)
	devices = devices[:num_workers]

	# Spawn one worker process per device.
	result_queue = mp.get_context('spawn').SimpleQueue()
	mp.spawn(test_worker, args=(devices, model_params, num_acc_points, num_cascades, recon_resolution, sample_splits, result_queue), nprocs=num_workers, join=True)

	# Compute average results.
	total_recon_loss = 0
	total_chamfer_dist = 0
	total_earth_dist = 0
	total_skipped_samples = 0

	for _ in range(num_workers):
		recon_loss, chamfer_dist, earth_dist, skipped_samples = result_queue.get()
		total_recon_loss += recon_loss
		total_chamfer_dist += chamfer_dist
		total_earth_dist += earth_dist
		total_skipped_samples += skipped_samples

	# Account for skipped samples.
	num_samples -= total_skipped_samples

	# Cover edge case where all mesh samples were bad.
	if num_samples == 0:
		return (None, None, None, total_skipped_samples)

	mean_recon_loss = total_recon_loss / num_samples
	mean_chamfer_dist = total_chamfer_dist / num_samples
	mean_earth_dist = total_earth_dist / num_samples

	return (mean_recon_loss, mean_chamfer_dist, mean_earth_dist, total_skipped_samples)


def get_test_paths(args):
	# Load all mesh files in test set directory and save to a list.
	if args.test_set_path:
		return get_mesh_files(args.test_set_path)
	# Load test files specified in the model output settings.
	elif args.train_output_path:
		test_set_names = get_test_set(args.train_output_path)
		return find_file_paths(args.mesh_data_dir, test_set_names)
	# Load files from mesh_data_dir.
	elif args.mesh_data_dir:
		# Load test set file names from the trained model parameters.
		torch.serialization.add_safe_globals([argparse.Namespace, Subset, RotationAxis, timedelta])
		save_data = torch.load(args.model_params, weights_only=True, map_location='cpu')
		data_splits = save_data['data_splits']
		test_split = data_splits[2] if data_splits is not None else None

		# Check that the provided model had a test set saved.
		if not test_split:
			print('The specified model file has no test set saved. Either provide the train_output_path or test_set_path arguments.')
			exit()

		# Load files from specified mesh directory.
		test_set_paths = get_mesh_files(args.mesh_data_dir)
		# Filter for files in the test set. Split entries are relative paths, so strip them to stems before comparing.
		test_split_names = set(os.path.splitext(os.path.basename(path))[0] for path in test_split)
		is_test_file = lambda path: os.path.splitext(os.path.basename(path))[0] in test_split_names
		return list(filter(is_test_file, test_set_paths))
	else:
		return None


def main():
	args = options()
	print('')

	# Parse devices.
	devices = get_devices(args.device, cpu_allowed=False)

	if args.train_output_path:
		args.model_params = os.path.join(args.train_output_path, BEST_MODEL_FILE)

	# Find test set mesh sample paths.
	test_set_paths = get_test_paths(args)

	# Validate test set samples.
	if test_set_paths is None:
		print('Failed to load test set files. Double check the provided arguments and try again.')
		exit()
	elif not test_set_paths:
		print('A test set was found but contains no valid samples. Double check the provided arguments and try again.')
		exit()

	# Test model.
	mean_recon_loss, mean_chamfer_dist, mean_earth_dist, total_skipped_samples = test(args.model_params, args.num_acc_points, args.num_cascades, args.recon_resolution, devices, test_set_paths)

	# Format result string.
	with io.StringIO() as str_out:
		print('', file=str_out)
		print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file=str_out)
		print(f'Number of Test Samples: {len(test_set_paths)}', file=str_out)
		print(f'Bad Samples Skipped:    {total_skipped_samples}', file=str_out)
		print('----------------------', file=str_out)
		print(f'Mean Reconstruction Loss:   {mean_recon_loss}', file=str_out)
		print(f'Mean Chamfer Distance:      {mean_chamfer_dist}', file=str_out)
		print(f'Mean Earth Movers Distance: {mean_earth_dist}', file=str_out)

		result_string = str_out.getvalue()

	# Print results to console.
	print(result_string)

	# Save result to file.
	if not args.result_file_path and args.train_output_path:
		args.result_file_path = os.path.join(args.train_output_path, TEST_RESULTS_FILE)

	if args.result_file_path:
		with open(args.result_file_path, 'a+') as f:
			f.write(result_string)


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
