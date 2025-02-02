import argparse
import numpy as np
import pyglet
import pyrender
import torch
import trimesh
import tkinter
from pyglet.gl import *
from utilities.csg_model import add_sdf
from utilities.csg_to_magica import prompt_and_export_to_magica
from utilities.csg_to_mesh import prompt_and_export_to_mesh
from utilities.file_loader import FileLoader


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

		colors = np.zeros(select_points.shape)
		colors[select_distances < 0, 2] = 1
		colors[select_distances > 0, 0] = 1
		cloud = pyrender.Mesh.from_points(select_points, colors=colors)
		self.mesh_node.mesh = cloud


	def load_samples(self, samples_file):
		samples = np.load(samples_file).astype(np.float32)

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
		self.file_loader = FileLoader(input_file)
		self.load_file(self.file_loader.get_file())

		super(SdfFileViewer, self).__init__(
			window_title,
			point_size,
			show_exterior_points,
			num_view_points)


	def load_file(self, samples_file):
		self.load_samples(samples_file)
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
	ORIGINAL_VIEW = "Original View"
	COMBINED_VIEW = "Combined View"
	RECON_VIEW = "Reconstruction View"


	def __init__(self, window_title, point_size, show_exterior_points, num_view_points, input_file, csg_model, sample_dist, get_csg_model=None):
		self.csg_model = csg_model
		self.num_view_points = num_view_points
		self.sample_dist = sample_dist
		self.get_csg_model = get_csg_model
		self.file_loader = FileLoader(input_file)
		self.load_samples(self.file_loader.get_file())
		self.view_mode = self.COMBINED_VIEW

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

		# Find default button based on current view mode
		if self.view_mode is self.ORIGINAL_VIEW:
			default_button = original_button
		elif self.view_mode is self.COMBINED_VIEW:
			default_button = combined_button
		elif self.view_mode is self.RECON_VIEW:
			default_button = reconstr_button

		# Create toggle button array
		self.add_button(ToggleButtons([original_button, combined_button, reconstr_button], default_button))

		export_x = self.viewport_size[0] - width - padding
		export_magica_y = padding
		export_mesh_y = export_magica_y + height + padding
		self.add_button(Button(export_x, export_mesh_y, width, height, 'Export to Mesh', callback=lambda: prompt_and_export_to_mesh(self.csg_model)))
		self.add_button(Button(export_x, export_magica_y, width, height, 'Export to MagicaCSG', callback=lambda: prompt_and_export_to_magica(self.csg_model)))


	def set_view_mode(self, mode):
		self.view_mode = mode
		self.update_mesh_node()


	def update_mesh_node(self):
		match self.view_mode:
			case self.ORIGINAL_VIEW:
				super(SdfModelViewer, self).update_mesh_node()
				return

			case self.COMBINED_VIEW:
				self.view_combined()
				return

			case self.RECON_VIEW:
				self.view_reconstruction()
				return


	def view_combined(self):
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


	def view_reconstruction(self):
		csg_view_points = self.num_view_points

		if csg_view_points < 0:
			csg_view_points = self.points.shape[0]

		# Sample reconstructed CSG model
		(csg_points, csg_distances) = self.csg_model.gen_csg_samples(1, csg_view_points)

		# Convert to numpy
		csg_points = csg_points.squeeze(0).cpu().numpy()
		csg_distances = csg_distances.squeeze(0).cpu().numpy()

		# Remove exterior points
		if not self.show_exterior_points:
			csg_points = csg_points[csg_distances <= 0, :]
			csg_distances = csg_distances[csg_distances <= 0]

		# Internal points are blue and external points are red
		colors = np.zeros(csg_points.shape)
		colors[csg_distances < 0, 2] = 1
		colors[csg_distances > 0, 0] = 1

		cloud = pyrender.Mesh.from_points(csg_points, colors=colors)
		self.mesh_node.mesh = cloud


	def load_samples(self, samples_file):
		self.csg_model = self.get_csg_model(samples_file)
		super(_SdfViewer, self).load_samples(samples_file)
		self.update_mesh_node()


	def view_prev(self):
		if self.get_csg_model == None:
			return

		samples_file = self.file_loader.prev_file()
		load_samples(samples_file)


	def view_next(self):
		if self.get_csg_model == None:
			return

		samples_file = self.file_loader.next_file()
		load_samples(samples_file)


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', required=True, type=str, help='Directory or Numpy file of sample points and distance values.')
	parser.add_argument('--num_view_points', type=int, default=-1, help='Number of points to display. Set to -1 to display all points.')
	parser.add_argument('--show_exterior_points', default=False, action='store_true', help='View points outside of the object.')
	parser.add_argument('--point_size', type=int, default=2, help='Size to render each point of the point cloud.')

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