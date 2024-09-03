import os
import argparse
import numpy as np
import trimesh
import pyrender
import numpy as np
import tkinter
from tkinter.constants import LEFT

LEFT_KEY = 65361
RIGHT_KEY = 65363

class SdfViewer(pyrender.Viewer):
	def __init__(self, input_file, num_view_points, point_size, show_exterior_points):
		self.input_file = input_file
		self.num_view_points = num_view_points
		self.point_size = point_size
		self.input_file = input_file

		(points, sdf) = self.load_samples(input_file, num_view_points)
		print(f'Point samples: {points.shape[0]}')
		window_title = "View: " + os.path.basename(input_file)

		if not show_exterior_points:
			points = points[sdf <= 0, :]
			sdf = sdf[sdf <= 0]

		colors = np.zeros(points.shape)
		colors[sdf < 0, 2] = 1
		colors[sdf > 0, 0] = 1
		cloud = pyrender.Mesh.from_points(points, colors=colors)
		scene = pyrender.Scene()
		self.mesh = cloud
		self.mesh_node = pyrender.Node(mesh=cloud)
		scene.add_node(self.mesh_node)

		super(SdfViewer, self).__init__(
			scene,
			use_raymond_lighting=True,
			point_size=point_size,
			show_world_axis=True,
			viewport_size=(1000,1000),
			window_title=window_title,
			view_center=[0,0,0])


	def load_samples(self, input_file, num_view_points):
		samples = np.load(input_file).astype(np.float32)

		if num_view_points > 0:
			samples = samples[:num_view_points,:]

		points = samples[:,:3]
		sdf = samples[:,3]

		return (points, sdf)


	def on_key_press(self, key, modifiers):
		if key == LEFT_KEY:
			print('left Key Pressed')
			self.mesh_node.mesh = pyrender.Mesh.from_points(np.array([1, 2, 3]))
		if key == RIGHT_KEY:
			print('right Key Pressed')
			self.mesh_node.mesh = self.mesh

		super(SdfViewer, self).on_key_press(key, modifiers)


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', required=True, type=str, help='Numpy file containing sample points and SDF values of input shape')
	parser.add_argument('--num_view_points', type=int, default=-1, help='Number of points to display. Set to -1 to display all points')
	parser.add_argument('--show_exterior_points', default=False, action='store_true', help='View points outside of the object')

	args = parser.parse_args()
	return args


def main():
	args = options()
	print('')
	viewer = SdfViewer(args.input_file, args.num_view_points, 2, args.show_exterior_points);


if __name__ == '__main__':
	main()