import argparse
import numpy as np
import torch
import trimesh
import pyrender
import numpy as np
import tkinter
from utilities.file_loader import FileLoader


class _SdfViewer(pyrender.Viewer):
	LEFT_KEY = 65361
	RIGHT_KEY = 65363

	def __init__(self, points, sdf, window_title, point_size, show_exterior_points):
		self.show_exterior_points = show_exterior_points
		self.mesh_node = pyrender.Node()
		self.set_points(points, sdf)

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


	def set_points(self, points, sdf):
		if not self.show_exterior_points:
			points = points[sdf <= 0, :]
			sdf = sdf[sdf <= 0]

		colors = np.zeros(points.shape)
		colors[sdf < 0, 2] = 1
		colors[sdf > 0, 0] = 1
		cloud = pyrender.Mesh.from_points(points, colors=colors)
		self.mesh_node.mesh = cloud


class SdfFileViewer(_SdfViewer):
	def __init__(self, input_file, num_view_points, point_size, show_exterior_points, window_title):
		self.num_view_points = num_view_points
		self.file_loader = FileLoader(input_file)
		(points, sdf) = self.load_file(input_file)

		super(SdfFileViewer, self).__init__(
			points,
			sdf,
			window_title,
			point_size,
			show_exterior_points)


	def load_file(self, input_file):
		self.input_file = input_file

		(points, sdf) = self.load_samples(input_file)
		print(f'Point samples: {points.shape[0]}')

		return (points, sdf);


	def load_samples(self, input_file):
		samples = np.load(input_file).astype(np.float32)

		if self.num_view_points > 0:
			samples = samples[:self.num_view_points,:]

		points = samples[:,:3]
		sdf = samples[:,3]

		return (points, sdf)


	def view_prev(self):
		file_path = self.file_loader.prev_file()
		self.set_points(*self.load_file(file_path))


	def view_next(self):
		file_path = self.file_loader.next_file()
		self.set_points(*self.load_file(file_path))


class SdfModelViewer(_SdfViewer):
	def __init__(self, csg_model, num_view_points, view_sampling, sample_dist, point_size, show_exterior_points, window_title,  get_prev_csg_model=None, get_next_csg_model=None):
		self.num_view_points = num_view_points
		self.sample_dist = sample_dist
		self.view_sampling = view_sampling
		self.num_view_points = num_view_points
		self.get_prev_csg_model = get_prev_csg_model
		self.get_next_csg_model = get_next_csg_model
		(points, sdf) = self.sampleCsg(csg_model)

		super(SdfModelViewer, self).__init__(
			points,
			sdf,
			window_title,
			point_size,
			show_exterior_points)


	def sampleCsg(self, csg_model):
		if self.view_sampling == 'uniform':
			(points, sdf) = csg_model.sample_csg_uniform(1, self.num_view_points)
		else:
			(points, sdf) = csg_model.sample_csg_surface(1, self.num_view_points, self.sample_dist)

		points = points.to(torch.device('cpu')).squeeze(0)
		sdf = sdf.to(torch.device('cpu')).squeeze(0)
		return (points, sdf)


	def view_prev(self):
		if self.get_prev_csg_model == None:
			return

		csg_model = self.get_prev_csg_model()
		self.set_points(*self.sampleCsg(csg_model))


	def view_next(self):
		if self.get_next_csg_model == None:
			return

		csg_model = self.get_next_csg_model()
		self.set_points(*self.sampleCsg(csg_model))


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
	viewer = SdfFileViewer(args.input_file, args.num_view_points, 2, args.show_exterior_points, "View SDF");


if __name__ == '__main__':
	main()