import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform

from utilities.data_processing import *
from utilities.datasets import PointDataset

from networks.csg_crn import CSG_CRN
from utilities.sdf_csg import CSGModel
from losses.loss import Loss


DATA_SPLIT = [0.8, 0.2]
# Number of options for selecting primitives or operations
PRIMITIVES_SIZE = 3
OPERATIONS_SIZE = 2
# Weights for regressor functions
PRIMITIVE_WEIGHT = 0.01
SHAPE_WEIGHT = 0.01
OPERATION_WEIGHT = 0.01


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	# Data settings
	parser.add_argument('--data_dir', type=str, required=True, help='Dataset parent directory')
	parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for checkpoints, trained model, and augmented dataset')
	parser.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')

	# Model settings
	parser.add_argument('--no_preprocess', default=False, action='store_true', help='Disable near-surface sample preprocessing')
	parser.add_argument('--clamp_dist', type=float, default=0.1, help='How close to the surface to sample')
	parser.add_argument('--num_input_points', type=int, default=1024, help='Number of points in the input point clouds')
	parser.add_argument('--num_loss_points', type=int, default=20000, help='Number of points to use in computing the loss')
	parser.add_argument('--num_prims', type=int, default=10, help='Number of primitives to generate each iteration')
	parser.add_argument('--num_iters', type=int, default=5, help='Number refinement iterations to train for')
	parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
	parser.add_argument('--max_epochs', type=int, default=2000, help='Maximum number of epochs to train')

	# Training settings
	parser.add_argument('--num_workers', type=int, default=8, help='Number of processes spawned for data loader')
	parser.add_argument('--device', type=str, default='', help='Select preffered training device')

	args = parser.parse_args()
	return args


# Determine device to train on
def get_device(device):
	if device:
		device = torch.device(device)

	elif torch.cuda.is_available():
		device = torch.device('cuda')

	else:
		device = torch.device('cpu')

	return device


# Prepare data files and load training dataset
def load_train_set(data_dir, output_path, no_preprocess, clamp_dist, num_input_points, num_loss_points, data_split):
	# Load sample files
	filenames = get_data_files(data_dir)
	print('Found %i data files' % len(filenames))

	# Create near-surface sample files
	if not no_preprocess:
		print('Selecting near-surface points...')
		data_dir = uniform_to_surface_data(data_dir, filenames, output_path, clamp_dist)

	# Split dataset and save to file
	train_files, test_files = torch.utils.data.random_split(filenames, data_split)
	save_list(os.path.join(output_path, 'train.txt'), train_files)
	save_list(os.path.join(output_path, 'test.txt'), test_files)

	print('Iniitalizing dataset...')
	train_dataset = PointDataset(data_dir, filenames, num_input_points, num_loss_points)
	return train_dataset


# Iteratively predict primitives and propagate average loss
def train_one_epoch(model, loss, optimizer, train_loader):
	model.train(True)

	for (index, data) in enumerate(train_loader):
		# Load data
		(target_all_samples, target_select_samples) = data
		target_all_points = target_all_samples[..., :3]
		target_all_distances = target_all_samples[..., 3]

		# Generate random samples for empty initial reconstruction input
		(batch_size, num_select_points, _) = target_select_samples.size()
		initial_samples = Uniform(-0.5, 0.5).sample((batch_size, num_select_points, 3))
		ones = torch.ones(batch_size, num_select_points, 1)
		initial_samples = torch.cat((initial_samples, ones), dim=-1)

		# Set sample distances for empty initial reconstruction loss
		(batch_size, num_all_points, _) = target_all_points.size()
		initial_distances = torch.ones(batch_size, num_all_points)

		# Initialize SDF CSG model
		csg_model = CSGModel(device)

		# Send all data to training device
		target_all_points = target_all_points.to(device)
		target_all_distances = target_all_distances.to(device)
		target_select_samples = target_select_samples.to(device)
		initial_samples = initial_samples.to(device)
		initial_distances = initial_distances.to(device)

		# Predict next primitive
		outputs = model(target_select_samples, initial_samples)

		# Sample predicted primitive
		csg_model.add_command(*outputs)
		refined_distances = csg_model.sample_csg(target_all_points)

		# Compute loss
		batch_loss = loss(target_all_distances, initial_distances, refined_distances, outputs[0], outputs[1])
		print(batch_loss)

		# Back propagate
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()


if __name__ == '__main__':
	args = options()

	# Set training device
	device = get_device(args.device)
	torch.multiprocessing.set_start_method('spawn')

	# Load training set
	output_path = create_out_dir(args)
	train_set = load_train_set(args.data_dir, output_path, args.no_preprocess, args.clamp_dist, args.num_input_points, args.num_loss_points, DATA_SPLIT)
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

	# Train model
	model = CSG_CRN(PRIMITIVES_SIZE, OPERATIONS_SIZE).to(device)
	loss = Loss(args.clamp_dist, PRIMITIVE_WEIGHT, SHAPE_WEIGHT, OPERATION_WEIGHT).to(device)
	torch.autograd.set_detect_anomaly(True)
	optimizer = torch.optim.Adam(model.parameters())
	train_one_epoch(model, loss, optimizer, train_loader)