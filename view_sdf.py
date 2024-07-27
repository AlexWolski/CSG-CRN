import os
import argparse
import numpy as np
import trimesh
import pyrender
import numpy as np

# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', required=True, type=str, help='Numpy file containing sample points and SDF values of input shape')
	parser.add_argument('--num_view_points', type=int, default=-1, help='Number of points to display. Set to -1 to display all points')
	parser.add_argument('--show_exterior_points', default=False, action='store_true', help='View points outside of the object')

	args = parser.parse_args()
	return args


def get_filename(args):
	return os.path.basename(args.input_file)


def load_samples(args):
	samples = np.load(args.input_file).astype(np.float32)

	if args.num_view_points > 0:
		samples = samples[:args.num_view_points,:]

	points = samples[:,:3]
	sdf = samples[:,3]

	return (points, sdf)


def display_points(points, sdf, window_title, point_size, show_exterior_points):
	if not show_exterior_points:
		points = points[sdf <= 0, :]
		sdf = sdf[sdf <= 0]

	colors = np.zeros(points.shape)
	colors[sdf < 0, 2] = 1
	colors[sdf > 0, 0] = 1
	cloud = pyrender.Mesh.from_points(points, colors=colors)
	scene = pyrender.Scene()
	scene.add(cloud)
	viewer = pyrender.Viewer(
		scene,
		use_raymond_lighting=True,
		point_size=point_size,
		show_world_axis=True,
		viewport_size=(1000,1000),
		window_title=window_title,
		view_center=[0,0,0])


def main():
	args = options()
	print('')

	(points, sdf) = load_samples(args)
	print(f'Point samples: {points.shape[0]}')
	window_title = "View: " + get_filename(args)
	display_points(points, sdf, window_title, 2, args.show_exterior_points)


if __name__ == '__main__':
	main()