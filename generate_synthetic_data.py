import argparse
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from utilities import sampler_utils
from utilities.data_augmentation import RotationAxis, ScaleAxis, random_rotation_batch, random_scale_batch
from utilities.point_transform import center_point_cloud_batch
from utilities.csg_model import CSGModel, add_sdf


# Number of samples to take form a CSG model to find a near-surface point
POSITION_SAMPLE_POINTS = 2000


# Parse commandline arguments
def options():
	# Parsers
	help_parser = argparse.ArgumentParser(add_help=False)
	data_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	gen_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
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
	gen_group.add_argument('--surface_uniform_ratio', type=float, default=0.5, help='Percentage of near-surface samples to select. 0 for only uniform samples and 1 for only near-surface samples')
	gen_group.add_argument('--sample_dist', type=float, default=0.1, help='Maximum distance to object surface for near-surface sampling (must be >0)')
	gen_group.add_argument('--min_scale', type=float, default=0.2, help='Lower bound on random scale value')
	gen_group.add_argument('--max_scale', type=float, default=0.8, help='Upper bound on random scale value')
	gen_group.add_argument('--no_blending', default=False, action='store_true', help='Disable primitive blending')
	gen_group.add_argument('--min_blending', type=float, default=0.0, help='Minimum bound for random primitive blending value')
	gen_group.add_argument('--max_blending', type=float, default=0.2, help='Maximum bound for random primitive blending value')
	gen_group.add_argument('--no_roundness', default=False, action='store_true', help='Disable primitive rounding')
	gen_group.add_argument('--min_roundness', type=float, default=0.0, help='Minimum bound for random roundness value')
	gen_group.add_argument('--max_roundness', type=float, default=0.2, help='Maximum bound for random roundness value')

	# Parse and handle Help argument
	args, remaining_args = help_parser.parse_known_args()

	if args.help or not remaining_args:
		print()
		data_parser.print_help()
		print('\n')
		gen_parser.print_help()
		exit()

	# Parse settings
	args, remaining_args = data_parser.parse_known_args(args=remaining_args, namespace=args)
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
	points = sampler_utils.sample_uniform_points_cube(POSITION_SAMPLE_POINTS, batch_size=1).to(csg_model.device)
	distances = csg_model.sample_csg(points)

	# Return the origin if the CSG model contains no shapes
	if distances is None:
		return torch.zeros(1, 3, dtype=float)

	# Find and return the point closest to the CSG model surface
	min_dist_index = torch.argmin(distances, dim=1)
	return points.squeeze()[min_dist_index]


# Generate a tensor with random float value within the given bounds
def rand_float_tensor(min_bound, max_bound, device):
	return torch.Tensor([[random.uniform(min_bound,max_bound)]]).to(device) 


# Randomly generate a shape to add to the CSG model
def generate_shape(csg_model, is_first_shape, no_blending, min_blending, max_blending, no_roundness, min_roundness, max_roundness, min_scale, max_scale):
	shape_weights = random_shape().to(csg_model.device)
	operation_weights = random_operation(is_first_shape).to(csg_model.device)
	translation = random_position(csg_model).to(csg_model.device)
	rotation = random_rotation_batch(RotationAxis.allAxes, 1).to(csg_model.device)
	scale = random_scale_batch(ScaleAxis.allAxes, min_scale, max_scale, 1).to(csg_model.device)
	no_blending = rand_float_tensor(min_blending, max_blending, csg_model.device) if not (no_blending or is_first_shape) else None
	no_roundness = rand_float_tensor(min_roundness, max_roundness, csg_model.device) if not no_roundness else None
	csg_model.add_command(shape_weights, operation_weights, translation, rotation, scale, no_blending, no_roundness)


# Generate a synthetic dataset of CSG model samples with random shapes and operations
def generate_dataset(args):
	os.makedirs(args.output_dir, exist_ok=True)

	for i in tqdm(range(args.num_samples)):
		output_path = os.path.join(args.output_dir, f'Sample {i}.npy')

		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		csg_model = CSGModel(1, device)

		# Generate model
		for i in range(args.num_shapes):
			is_first_shape = i == 0
			generate_shape(csg_model, is_first_shape, args.no_blending, args.min_blending, args.max_blending, args.no_roundness, args.min_roundness, args.max_roundness, args.min_scale, args.max_scale)

		# Sample model
		samples = sampler_utils.sample_sdf_from_csg_combined(csg_model, args.num_sample_points, args.sample_dist, args.surface_uniform_ratio)

		# Re-generate sample if there are no samples
		if samples is None:
			i = i-1
			continue

		# Center sample
		(sample_points, sample_distances) = samples
		sample_points = center_point_cloud_batch(sample_points)

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