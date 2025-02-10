import argparse
import numpy as np
import pyglet
import torch
import trimesh
import tkinter

from pyglet.gl import *
from utilities.csg_model import add_sdf
from utilities.csg_to_magica import prompt_and_export_to_magica
from utilities.csg_to_mesh import prompt_and_export_to_mesh
from utilities.file_loader import FileLoader
from utilities.sampler_utils import sample_sdf_from_mesh_unit_sphere
from utilities.file_utils import MESH_FILE_TYPES
from utilities.sampler_utils import sample_points_mesh_surface, distance_to_mesh_surface

import pyrender


class Button():
	def __init__(self, x, y, width, height, text='', text_color=(0,0,0), color=(200,200,200), pressed_color=(100,100,100), callback=None):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.text = text
		self.text_color = text_color
		self.color = color
		self.pressed_color = pressed_color
		self.callback = callback
		self.pressed = False
		self.focused = False
		self.disabled = False


	def draw(self):
		if self.pressed:
			button_color = self.pressed_color
		else:
			button_color = self.color

		rect = pyglet.shapes.Rectangle(self.x, self.y, self.width, self.height, button_color)
		text_x = self.x + self.width // 2
		text_y = self.y + self.height // 2
		text = pyglet.text.Label(self.text, x=text_x, y=text_y, anchor_x='center', anchor_y='center', color=self.text_color)

		glDisable(GL_DEPTH_TEST)
		rect.draw()
		text.draw()


	def disable(self):
		self.disabled = True
		self.pressed = True


	def enable(self):
		self.disabled = False
		self.pressed = False


	def is_over_button(self, mouse_x, mouse_y):
		return (self.x < mouse_x < self.x + self.width and
				self.y < mouse_y < self.y + self.height)


	def on_mouse_press(self, mouse_x, mouse_y):
		if self.disabled:
			return

		if self.is_over_button(mouse_x, mouse_y):
			self.pressed = True
			self.focused = True


	def on_mouse_drag(self, mouse_x, mouse_y):
		if self.disabled:
			return

		self.pressed = self.focused and self.is_over_button(mouse_x, mouse_y)


	def on_mouse_release(self, mouse_x, mouse_y):
		if self.disabled:
			return

		if self.focused and self.is_over_button(mouse_x, mouse_y) and self.callback:
			self.callback()

		self.pressed = False
		self.focused = False


class ToggleButtons():
	def __init__(self, buttons, default_button):
		if default_button not in set(buttons):
			raise Exception('The provided buttons list must contain default_button')

		self.buttons = buttons

		for button in self.buttons:
			if button == default_button:
				button.disable()
			else:
				button.enable()


	def draw(self):
		for button in self.buttons:
			button.draw()


	def on_mouse_press(self, mouse_x, mouse_y):
		for button in self.buttons:
			button.on_mouse_press(mouse_x, mouse_y)


	def on_mouse_drag(self, mouse_x, mouse_y):
		for button in self.buttons:
			button.on_mouse_drag(mouse_x, mouse_y)


	def on_mouse_release(self, mouse_x, mouse_y):
		pressed_button = None

		# Find button that was pressed
		for button in self.buttons:
			if button.is_over_button(mouse_x, mouse_y):
				pressed_button = button

		# No action if the button is already depressed
		if pressed_button is None or pressed_button.disabled:
			return

		# Unpress all buttons
		for button in self.buttons:
			button.enable()

		# Trigger clicked button
		pressed_button.callback()
		pressed_button.disable()


class _SdfViewer(pyrender.Viewer):
	LEFT_KEY = 65361
	RIGHT_KEY = 65363
	buttons = []

	def __init__(self, window_title, point_size, show_exterior_points, num_view_points, view_width=1000, view_height=1000):
		self.show_exterior_points = show_exterior_points
		self.num_view_points = num_view_points
		self.mesh_node = pyrender.Node()
		self.update_mesh_node()
		self.view_width = view_width
		self.view_height = view_height

		scene = pyrender.Scene()
		scene.add_node(self.mesh_node)

		super(_SdfViewer, self).__init__(
			scene,
			use_raymond_lighting=True,
			point_size=point_size,
			show_world_axis=True,
			viewport_size=(self.view_width, self.view_height),
			window_title=window_title,
			view_center=[0,0,0],
			run_in_thread=True)


	def add_button(self, button):
		self.buttons.append(button)


	def set_points(self, points, distances):
		self.points = points
		self.distances = distances


	def update_mesh_node(self):
		select_points = self.points
		select_distances = self.distances

		if not self.show_exterior_points:
			select_points = select_points[self.distances <= 0, :]
			select_distances = select_distances[self.distances <= 0]

		colors = torch.zeros(select_points.size())
		colors[select_distances < 0, 2] = 1
		colors[select_distances > 0, 0] = 1
		cloud = pyrender.Mesh.from_points(select_points.cpu().numpy(), colors=colors)
		self.mesh_node.mesh = cloud


	def load_samples(self, samples):
		if len(samples.size()) > 2:
			raise Exception(f'Expected SDF sample tensor with two dimensions but got {len(samples.size())}.')

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


	def on_mouse_press(self, x, y, button, modifiers):
		for button in self.buttons:
			button.on_mouse_press(x, y)

		super(_SdfViewer, self).on_mouse_press(x, y, button, modifiers)


	def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
		for button in self.buttons:
			button.on_mouse_drag(x, y)

		super(_SdfViewer, self).on_mouse_drag(x, y, dx, dy, buttons, modifiers)


	def on_mouse_release(self, x, y, button, modifiers):
		for button in self.buttons:
			button.on_mouse_release(x, y)

		super(_SdfViewer, self).on_mouse_release(x, y, button, modifiers)


	def on_draw(self):
		super(_SdfViewer, self).on_draw()

		for button in self.buttons:
			button.draw()


	def view_prev(self):
		raise NotImplementedError("Implement method in inheriting class")


	def view_next(self):
		raise NotImplementedError("Implement method in inheriting class")


class SdfFileViewer(_SdfViewer):
	def __init__(self, window_title, point_size, show_exterior_points, num_view_points, input_file):
		self.num_view_points = num_view_points
		self.file_loader = FileLoader(input_file, ['*.npy'])
		self.load_file(self.file_loader.get_file())

		super(SdfFileViewer, self).__init__(
			window_title,
			point_size,
			show_exterior_points,
			num_view_points)


	def load_file(self, samples_file):
		samples_numpy = np.load(samples_file).astype(np.float32)
		samples = torch.from_numpy(samples_numpy)
		self.load_samples(samples)
		print(f'Point samples: {self.points.size(0)}')


	def view_prev(self):
		file_path = self.file_loader.prev_file()
		self.load_file(file_path)
		self.update_mesh_node()


	def view_next(self):
		file_path = self.file_loader.next_file()
		self.load_file(file_path)
		self.update_mesh_node()


class SdfModelViewer(_SdfViewer):
	ORIGINAL_VIEW = "Original View"
	COMBINED_VIEW = "Combined View"
	RECON_VIEW = "Reconstruction View"

	POINT_MODEL = "Point Cloud Model"
	MESH_MODEL = "Mesh Model"


	def __init__(self, window_title, point_size, show_exterior_points, num_view_points, input_file, sample_dist, get_mesh_and_csg_model=None):
		self.view_mode = self.COMBINED_VIEW
		self.model_mode = self.POINT_MODEL
		self.num_view_points = num_view_points
		self.get_mesh_and_csg_model = get_mesh_and_csg_model
		self.file_loader = FileLoader(input_file, MESH_FILE_TYPES)
		self.sample_dist = sample_dist
		self.load_file(self.file_loader.get_file())

		super(SdfModelViewer, self).__init__(
			window_title,
			point_size,
			show_exterior_points,
			num_view_points)

		self._init_buttons()


	def _init_buttons(self):
		padding = 10
		width = 200
		height = 50

		recon_y = padding
		combined_y = recon_y + height + padding
		original_y = combined_y + height + padding

		original_button = Button(padding, original_y, width, height, 'View Original', callback=lambda: self.set_view_mode(self.ORIGINAL_VIEW))
		combined_button = Button(padding, combined_y, width, height, 'View Combined', callback=lambda: self.set_view_mode(self.COMBINED_VIEW))
		reconstr_button = Button(padding, padding, width, height, 'View Reconstruction', callback=lambda: self.set_view_mode(self.RECON_VIEW))

		# Find default view button based on current view mode
		if self.view_mode is self.ORIGINAL_VIEW:
			default_view_button = original_button
		elif self.view_mode is self.COMBINED_VIEW:
			default_view_button = combined_button
		elif self.view_mode is self.RECON_VIEW:
			default_view_button = reconstr_button

		# Create view button toggle array
		self.add_button(ToggleButtons([original_button, combined_button, reconstr_button], default_view_button))

		mesh_y = original_y + 120
		point_y = mesh_y + height + padding

		point_button = Button(padding, point_y, width, height, 'Point Cloud', callback=lambda: self.set_model_mode(self.POINT_MODEL))
		mesh_button = Button(padding, mesh_y, width, height, 'Mesh', callback=lambda: self.set_model_mode(self.MESH_MODEL))

		# Find default model button based on current model mode
		if self.model_mode is self.MESH_MODEL:
			default_model_button = mesh_button
		elif self.model_mode is self.POINT_MODEL:
			default_model_button = point_button

		# Create model button toggle array
		self.add_button(ToggleButtons([mesh_button, point_button], default_model_button))

		# Create export buttons
		export_x = self.viewport_size[0] - width - padding
		export_magica_y = padding
		export_mesh_y = export_magica_y + height + padding
		self.add_button(Button(export_x, export_mesh_y, width, height, 'Export to Mesh', callback=lambda: prompt_and_export_to_mesh(self.csg_model)))
		self.add_button(Button(export_x, export_magica_y, width, height, 'Export to MagicaCSG', callback=lambda: prompt_and_export_to_magica(self.csg_model)))


	def set_view_mode(self, mode):
		self.view_mode = mode
		self.update_mesh_node()


	def set_model_mode(self, mode):
		self.model_mode = mode
		self.update_mesh_node()


	def update_mesh_node(self):
		match self.model_mode:
			case self.POINT_MODEL:
				self.update_point_cloud()

			case self.MESH_MODEL:
				self.update_mesh()
				return


	def update_point_cloud(self):
		match self.view_mode:
			case self.ORIGINAL_VIEW:
				self.view_point_cloud(self.target_surface_points)
				return

			case self.COMBINED_VIEW:
				self.view_combined_point_cloud()
				return

			case self.RECON_VIEW:
				self.view_point_cloud(self.recon_surface_points)
				return


	def update_mesh(self):
		match self.view_mode:
			case self.ORIGINAL_VIEW:
				self.mesh_node.mesh = pyrender.Mesh.from_trimesh(self.target_mesh)
				return

			case self.COMBINED_VIEW:
				return

			case self.RECON_VIEW:
				self.mesh_node.mesh = pyrender.Mesh.from_trimesh(self.recon_mesh)
				return


	def view_combined_point_cloud(self):
		# Compute distance from target surface points to CSG reconstruction
		target_to_recon_distances = self.csg_model.sample_csg(self.target_surface_points.to(self.csg_model.device).unsqueeze(0)).squeeze(0)

		# Color overlapping points purple and target points outside of the CSG reconstruction blue
		target_colors = torch.zeros(self.target_surface_points.size())
		target_colors[target_to_recon_distances < 0, 0] = 1
		target_colors[:,2] = 1

		# Compute distance from reconstruction surface points to target mesh
		recon_to_target_distances = distance_to_mesh_surface(self.target_mesh, self.recon_surface_points)

		# Color overlapping points purple and reconstruction points outside of the target mesh red
		recon_colors = torch.zeros(self.recon_surface_points.size())
		recon_colors[:,0] = 1
		recon_colors[recon_to_target_distances < 0, 2] = 1

		# Combine target and reconstruction points
		all_points = torch.cat((self.target_surface_points, self.recon_surface_points), dim=0)
		all_colors = np.concatenate((target_colors, recon_colors), axis=0)

		cloud = pyrender.Mesh.from_points(all_points.cpu().numpy(), colors=all_colors)
		self.mesh_node.mesh = cloud


	def view_point_cloud(self, surface_points):
		# Sample reconstructed CSG model
		colors = np.array([[0, 0, 255]]).repeat(self.num_view_points, axis=0)
		cloud = pyrender.Mesh.from_points(surface_points.squeeze(0).cpu().numpy(), colors=colors)
		self.mesh_node.mesh = cloud


	def load_file(self, samples_file):
		self.target_mesh, self.recon_mesh, self.csg_model = self.get_mesh_and_csg_model(samples_file)
		self.target_surface_points = sample_points_mesh_surface(self.target_mesh, self.num_view_points)
		self.recon_surface_points = sample_points_mesh_surface(self.recon_mesh, self.num_view_points)


	def view_prev(self):
		if self.get_mesh_and_csg_model == None:
			return

		samples_file = self.file_loader.prev_file()
		self.load_file(samples_file)
		self.update_mesh_node()


	def view_next(self):
		if self.get_mesh_and_csg_model == None:
			return

		samples_file = self.file_loader.next_file()
		self.load_file(samples_file)
		self.update_mesh_node()


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', required=True, type=str, help='Directory or Numpy file of sample points and distance values.')
	parser.add_argument('--num_view_points', type=int, default=-1, help='Number of points to display. Set to -1 to display all points.')
	parser.add_argument('--show_exterior_points', default=False, action='store_true', help='View points outside of the object.')
	parser.add_argument('--point_size', type=int, default=3, help='Size to render each point of the point cloud.')

	args = parser.parse_args()
	return args


def main():
	args = options()
	print('')

	try:
		viewer = SdfFileViewer("View SDF", args.point_size, args.show_exterior_points, args.num_view_points, args.input_file)
	except FileNotFoundError as fileError:
		print(fileError)
	except Exception:
		print(traceback.format_exc())



if __name__ == '__main__':
	main()