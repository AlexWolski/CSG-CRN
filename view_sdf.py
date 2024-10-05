import os
import argparse
import numpy as np
import trimesh
import pyrender
import numpy as np
import tkinter
from pathlib import Path


LEFT_KEY = 65361
RIGHT_KEY = 65363


class SdfViewer(pyrender.Viewer):
	def __init__(self, input_file, num_view_points, point_size, show_exterior_points):
		self.num_view_points = num_view_points
		self.show_exterior_points = show_exterior_points
		self.mesh_node = pyrender.Node()

		self.set_file(input_file)
		self.load_file_names(input_file)

		scene = pyrender.Scene()
		scene.add_node(self.mesh_node)
		window_title = "View: " + os.path.basename(input_file)

		super(SdfViewer, self).__init__(
			scene,
			use_raymond_lighting=True,
			point_size=point_size,
			show_world_axis=True,
			viewport_size=(1000,1000),
			window_title=window_title,
			view_center=[0,0,0])


	def set_file(self, input_file):
		self.input_file = input_file

		(points, sdf) = self.load_samples(input_file)
		print(f'Point samples: {points.shape[0]}')

		if not self.show_exterior_points:
			points = points[sdf <= 0, :]
			sdf = sdf[sdf <= 0]

		colors = np.zeros(points.shape)
		colors[sdf < 0, 2] = 1
		colors[sdf > 0, 0] = 1
		cloud = pyrender.Mesh.from_points(points, colors=colors)
		self.mesh_node.mesh = cloud


	def load_samples(self, input_file):
		samples = np.load(input_file).astype(np.float32)

		if self.num_view_points > 0:
			samples = samples[:self.num_view_points,:]

		points = samples[:,:3]
		sdf = samples[:,3]

		return (points, sdf)


	def load_file_names(self, input_file):
		self.file_list = []

		input_file_path = Path(input_file)
		input_file_name = input_file_path.name
		self.parent_dir = input_file_path.parent.absolute()
		index = 0

		for file in os.listdir(self.parent_dir):
			if file.endswith(".npy"):
				if file == input_file_name:
					self.file_index = index

				self.file_list.append(file)
				index += 1


	def load_next_file(self):
		self.file_index += 1

		if self.file_index >= len(self.file_list):
			self.file_index = 0;

		file_name = self.file_list[self.file_index]
		file_path = self.parent_dir.joinpath(file_name)

		self.set_file(file_path)


	def load_prev_file(self):
		self.file_index -= 1

		if self.file_index < 0:
			self.file_index = len(self.file_list) - 1;

		file_name = self.file_list[self.file_index]
		file_path = self.parent_dir.joinpath(file_name)

		self.set_file(file_path)


	def on_key_press(self, key, modifiers):
		if key == LEFT_KEY:
			self.load_prev_file()
		if key == RIGHT_KEY:
			self.load_next_file()

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