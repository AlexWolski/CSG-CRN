import argparse
import numpy as np
import torch
import trimesh
import pyrender
import numpy as np
import tkinter
from utilities.file_loader import FileLoader
from utilities.csg_model import add_sdf


class _SdfViewer(pyrender.Viewer):
	LEFT_KEY = 65361
	RIGHT_KEY = 65363

	def __init__(self, window_title, point_size, show_exterior_points, num_view_points):
		self.show_exterior_points = show_exterior_points
		self.num_view_points = num_view_points
		self.mesh_node = pyrender.Node()
		self.update_mesh_node()

		scene = pyrender.Scene()
		scene.add_node(self.mesh_node)

		super(_SdfViewer, self).__init__(
			scene,
			use_raymond_lighting=True,
			point_size=point_size,
			show_world_axis=True,
			viewport_size=(1000,1000),
			window_title=window_title,
			view_center=[0,0,0])


	def set_points(self, points, distances):
		self.points = points
		self.distances = distances


	def update_mesh_node(self):
		if not self.show_exterior_points:
			self.points = self.points[self.distances <= 0, :]
			self.distances = self.distances[self.distances <= 0]

		colors = np.zeros(self.points.shape)
		colors[self.distances < 0, 2] = 1
		colors[self.distances > 0, 0] = 1
		cloud = pyrender.Mesh.from_points(self.points, colors=colors)
		self.mesh_node.mesh = cloud


	def load_samples(self, input_file):
		samples = np.load(input_file).astype(np.float32)

		if self.num_view_points > 0:
			samples = samples[:self.num_view_points,:]

		points = samples[:,:3]
		distances = samples[:,3]

		self.set_points(points, distances)


	def on_key_press(self, key, modifiers):
		if key == self.LEFT_KEY:
			self.view_prev()
		if key == self.RIGHT_KEY:
			self.view_next()

		super(_SdfViewer, self).on_key_press(key, modifiers)


	def view_prev(self):
		raise NotImplementedError("Implement method in inheriting class")


	def view_next(self):
		raise NotImplementedError("Implement method in inheriting class")


class SdfFileViewer(_SdfViewer):
	def __init__(self, window_title, point_size, show_exterior_points, num_view_points, input_file):
		self.num_view_points = num_view_points
		self.file_loader = FileLoader(input_file)
		self.load_file(input_file)

		super(SdfFileViewer, self).__init__(
			window_title,
			point_size,
			show_exterior_points,
			num_view_points)


	def load_file(self, input_file):
		self.input_file = input_file
		self.load_samples(input_file)
		print(f'Point samples: {self.points.shape[0]}')


	def view_prev(self):
		file_path = self.file_loader.prev_file()
		self.load_file(file_path)
		self.update_mesh_node()


	def view_next(self):
		file_path = self.file_loader.next_file()
		self.load_file(file_path)
		self.update_mesh_node()


class SdfModelViewer(_SdfViewer):
	def __init__(self, window_title, point_size, show_exterior_points, num_view_points, input_file, csg_model, view_sampling, sample_dist, get_csg_model=None):
		self.csg_model = csg_model
		self.num_view_points = num_view_points
		self.sample_dist = sample_dist
		self.view_sampling = view_sampling
		self.get_csg_model = get_csg_model
		self.file_loader = FileLoader(input_file)
		self.load_samples(input_file)

		super(SdfModelViewer, self).__init__(
			window_title,
			point_size,
			show_exterior_points,
			num_view_points)


	def update_mesh_node(self):
		# Convert numpy arrays to torch tensors
		torch_points = torch.from_numpy(self.points).to(self.csg_model.device)
		torch_distances = torch.from_numpy(self.distances).to(self.csg_model.device)

		# Get distances from sample points to csg model
		csg_distances = self.csg_model.sample_csg(torch_points.unsqueeze(0)).squeeze(0)

		# Apply union on distances of original shape and reconstruction
		combined_sdf = add_sdf(torch_distances, csg_distances, None).squeeze(0)

		# Convert to numpy again
		csg_distances = csg_distances.cpu().numpy()
		combined_sdf = combined_sdf.cpu().numpy()

		# Remove exterior points
		internal_points = self.points[combined_sdf <= 0, :]
		internal_distances = self.distances[combined_sdf <= 0]
		csg_distances = csg_distances[combined_sdf <= 0]

		# Original shape is red, reconstruction is blue, and the intersection is purple
		colors = np.zeros(internal_points.shape)
		colors[internal_distances < 0, 0] = 1
		colors[csg_distances < 0, 2] = 1

		cloud = pyrender.Mesh.from_points(internal_points, colors=colors)
		self.mesh_node.mesh = cloud


	def view_prev(self):
		if self.get_csg_model == None:
			return

		input_file = self.file_loader.prev_file()
		self.csg_model = self.get_csg_model(input_file)
		self.load_samples(input_file)
		self.update_mesh_node()


	def view_next(self):
		if self.get_csg_model == None:
			return

		input_file = self.file_loader.next_file()
		self.csg_model = self.get_csg_model(input_file)
		self.load_samples(input_file)
		self.update_mesh_node()


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', required=True, type=str, help='Numpy file containing sample points and SDF values of input shape')
	parser.add_argument('--num_view_points', type=int, default=-1, help='Number of points to display. Set to -1 to display all points')
	parser.add_argument('--show_exterior_points', default=False, action='store_true', help='View points outside of the object')
	parser.add_argument('--point_size', type=int, default=2, help='Size to render each point of the point cloud')

	args = parser.parse_args()
	return args


def main():
	args = options()
	print('')
	viewer = SdfFileViewer("View SDF", args.point_size, args.show_exterior_points, args.num_view_points, args.input_file)


if __name__ == '__main__':
	main()