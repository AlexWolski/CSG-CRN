import os
import argparse
import torch
from torch.utils.data import DataLoader

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
	parser.add_argument('--num_workers', type=int, default=8, help='Number of processes spawned for data loader')

	args = parser.parse_args()
	return args


# Prepare data files and load training dataset
def load_train_set(uniform_dir, output_path, sample_dist, num_points, data_split):
	# Load uniform sample files
	filenames = get_data_files(uniform_dir)
	print('Found %i data files' % len(filenames))

	# Create near-surface sample files
	print('Selecting near-surface points...')
	surface_dir = uniform_to_surface_data(uniform_dir, filenames, output_path, sample_dist)
	print('Done...')

	# Split dataset and save to file
	train_files, test_files = torch.utils.data.random_split(filenames, data_split)
	save_list(os.path.join(output_path, 'train.txt'), train_files)
	save_list(os.path.join(output_path, 'test.txt'), test_files)

	train_dataset = PointDataset(uniform_dir, surface_dir, filenames, num_points)
	return train_dataset


if __name__ == '__main__':
	args = options()

	# Load training set
	output_path = create_out_dir(args)
	train_set = load_train_set(args.data_dir, output_path, args.sample_dist, args.num_points, DATA_SPLIT)
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)