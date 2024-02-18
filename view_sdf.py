from mesh_to_sdf import sample_sdf_near_surface

import os
import argparse
import numpy as np
import trimesh
import pyrender
import numpy as np

# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--npy_file', required=True, type=str, help='Numpy file containing SDF samples')
	parser.add_argument('--num_points', type=int, default=-1, help='Number of points to display. Set to -1 to display all points')
	parser.add_argument('--exterior_points', default=False, action='store_true', help='View points outside of the object')

	args = parser.parse_args()
	return args


def get_filename(args):
	return os.path.basename(args.npy_file)


def load_samples(args):
	samples = np.load(args.npy_file).astype(np.float32)

	if not args.exterior_points:
		interior_sample_rows = np.where(samples[:,3] <= 0)
		samples = samples[interior_sample_rows]

	if args.num_points > 0:
		samples = samples[:args.num_points,:]

	points = samples[:,:3]
	sdf = samples[:,3]

	return (points, sdf)


def display_points(points, sdf, filename):
	colors = np.zeros(points.shape)
	colors[sdf < 0, 2] = 1
	colors[sdf > 0, 0] = 1
	cloud = pyrender.Mesh.from_points(points, colors=colors)
	scene = pyrender.Scene()
	scene.add(cloud)
	viewer = pyrender.Viewer(
		scene,
		use_raymond_lighting=True,
		point_size=2,
		show_world_axis=True,
		viewport_size=(1000,1000),
		window_title=filename,
		view_center=[0,0,0])


def main():
	args = options()
	print('')

	(points, sdf) = load_samples(args)
	filename = get_filename(args)
	display_points(points, sdf, filename)


if __name__ == '__main__':
	main()