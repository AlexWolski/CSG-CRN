import os
import argparse
import torch

from utilities.data_processing import *
from utilities.datasets import PointDataset


# Constant settings
DATA_SPLIT = [0.8, 0.2]


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', type=str, required=True, help='Dataset parent directory')
	parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for checkpoints, trained model, and augmented dataset')
	parser.add_argument('--sample_dist', type=float, default=0.1, help='How close to the surface to sample')
	parser.add_argument('--num_points', type=int, default=1024, help='Number of points in the input point clouds')
	parser.add_argument('--num_prims', type=int, default=10, help='Number of primitives to generate each iteration')
	parser.add_argument('--num_iters', type=int, default=5, help='Number refinement iterations to train for')
	parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
	parser.add_argument('--max_epochs', type=int, default=2000, help='Maximum number of epochs to train')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = options()

	output_path = create_out_dir(args)
	uniform_file_paths = get_data_files(args.data_dir)
	print('Found %i data files' % len(uniform_file_paths))

	print('Selecting near-surface points...')
	(surface_points_dir, surface_points_filenames) = uniform_to_surface_data(uniform_file_paths, args.sample_dist, output_path)
	print('Done...')

	surface_points_train, surface_points_test = torch.utils.data.random_split(surface_points_filenames, DATA_SPLIT)
	save_list(os.path.join(output_path, 'train.txt'), surface_points_train)
	save_list(os.path.join(output_path, 'test.txt'), surface_points_test)

	train_dataset = PointDataset(surface_points_dir, surface_points_train, args.num_points)