import argparse
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from utilities import data_augmentation
from utilities.data_augmentation import RotationAxis, ScaleAxis, random_rotation_batch, random_scale_batch, scale_to_unit_sphere_batch
from utilities.csg_model import CSGModel, add_sdf


# Number of samples to take form a CSG model to find a near-surface point
POSITION_SAMPLE_POINTS = 2000


# Parse commandline arguments
def options():
	# Parsers
	help_parser = argparse.ArgumentParser(add_help=False)
	data_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	gen_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	augment_parser = data_augmentation.get_augment_parser('OFFLINE AUGMENT SETTINGS')
	data_group = data_parser.add_argument_group('DATA SETTINGS')
	gen_group = gen_parser.add_argument_group('GENERATION SETTINGS')

	# Help flag
	help_parser.add_argument('-h', '--help', default=False, action='store_true', help='Print help text')

	# Data settings
	data_group.add_argument('--output_dir', type=str, default='./data/output', help='Output directory to store SDF samples')
	data_group.add_argument('--num_samples', type=int, default=10000, help='Number of SDF samples to compute')
	data_group.add_argument('--num_sample_points', type=int, default=10000, help='Number of points to sample for each SDF')
	data_group.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing files in output directory')

	# Shape generation parser
	gen_group.add_argument('--num_shapes', type=int, default=1, help='Number of shapes to generate per CSG model')
	gen_group.add_argument('--sample_method', default=['near-surface'], choices=['uniform', 'near-surface'], nargs=1, help='Select SDF samples uniformly or near object surfaces. Near-surface requires pre-processing')
	gen_group.add_argument('--sample_dist', type=float, default=0.001, help='Maximum distance to object surface for near-surface sampling (must be >0)')
	gen_group.add_argument('--no_blending', default=False, action='store_true', help='Disable primitive blending')
	gen_group.add_argument('--no_roundness', default=False, action='store_true', help='Disable primitive rounding')

	# Parse and handle Help argument
	args, remaining_args = help_parser.parse_known_args()

	if args.help or not remaining_args:
		print()
		data_parser.print_help()
		print('\n')
		augment_parser.print_help()
		exit()

	# Parse data settings
	args, remaining_args = data_parser.parse_known_args(args=remaining_args, namespace=args)

	# Parse augment settings
	args, remaining_args = augment_parser.parse_known_args(args=remaining_args, namespace=args)

	gen_parser.parse_args(args=remaining_args, namespace=args)

	return args


# Randomly select a shape
def random_shape():
	shape_index = random.randrange(CSGModel.num_shapes)
	shape_weights = torch.zeros(1, CSGModel.num_shapes, dtype=float)
	shape_weights[:,shape_index] = 1.0
	return shape_weights


# Randomly select a valid CSG operation
def random_operation(is_first_shape):
	if is_first_shape:
		op_index = CSGModel.operation_functions.index(add_sdf)
	else:
		op_index = random.randrange(CSGModel.num_operations)

	operation_weights = torch.zeros(1, CSGModel.num_operations, dtype=float)
	operation_weights[:,op_index] = 1.0
	return operation_weights


# Randomly sample a point near the surface of the CSG model
def random_position(csg_model):
	# Sample the CSG model
	sample_points = csg_model.sample_csg_uniform(1, POSITION_SAMPLE_POINTS)

	# Return the origin if the CSG model contains no shapes
	if not sample_points:
		return torch.zeros(1, 3, dtype=float)

	# Find and return the point closest to the CSG model surface
	(points, distances) = sample_points
	min_dist_index = torch.argmin(distances, dim=1)
	return points.squeeze()[min_dist_index]


# Randomly generate a shape to add to the CSG model
def generate_shape(csg_model, is_first_shape, no_blending, no_roundness, no_rotation, no_scale, rotate_axis, scale_axis, min_scale, max_scale):
	shape_weights = random_shape().to(csg_model.device)
	operation_weights = random_operation(is_first_shape).to(csg_model.device)
	translation = random_position(csg_model).to(csg_model.device)
	rotation = random_rotation_batch(rotate_axis, 1).to(csg_model.device)
	scale = random_scale_batch(scale_axis, min_scale, max_scale, 1).to(csg_model.device)
	no_blending = torch.rand(1,1).to(csg_model.device) if not (no_blending or is_first_shape) else None
	no_roundness = torch.rand(1,1).to(csg_model.device) if not no_roundness else None
	csg_model.add_command(shape_weights, operation_weights, translation, rotation, scale, no_blending, no_roundness)


# Generate a synthetic dataset of CSG model samples with random shapes and operations
def generate_dataset(args):
	os.makedirs(args.output_dir, exist_ok=True)

	for i in tqdm(range(args.num_samples)):
		is_first_shape = i == 0
		output_path = os.path.join(args.output_dir, f'Sample {i}.npy')

		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		csg_model = CSGModel(device)

		# Generate model
		for i in range(args.num_shapes):
			generate_shape(csg_model, is_first_shape, args.no_blending, args.no_roundness, args.no_rotation, args.no_scale, args.rotate_axis, args.scale_axis, args.min_scale, args.max_scale)

		# Sample model
		if args.sample_method[0] == 'uniform':
			(sample_points, sample_distances) = csg_model.sample_csg_uniform(1, args.num_sample_points)
		else:
			(sample_points, sample_distances) = csg_model.sample_csg_surface(1, args.num_sample_points, args.sample_dist)

		# Save model samples to file
		samples = torch.cat((sample_points, sample_distances.unsqueeze(-1)), dim=-1).squeeze(0).cpu()
		np.save(output_path, samples)


def main():
	args = options()
	print('')
	print(f'Processing {args.num_samples} samples...')
	generate_dataset(args)


if __name__ == '__main__':
	main()