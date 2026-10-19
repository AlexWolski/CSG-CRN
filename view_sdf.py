import argparse
import numpy as np
import pyglet
import torch

from pyglet.gl import *
from utilities.csg_model import CSGModel, add_sdf, get_operation_name
from utilities.csg_to_magica import prompt_and_export_to_magica
from utilities.csg_to_mesh import csg_to_mesh, prompt_and_export_to_mesh
from utilities.file_loader import FileLoader
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
			if button.focused and button.is_over_button(mouse_x, mouse_y):
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


class PrimitiveSlider():
	ADD_COLOR = (0, 180, 0)
	SUBTRACT_COLOR = (220, 0, 0)
	HIDDEN_ADD_COLOR = (150, 150, 150)
	UNSELECTED_ALPHA = 70
	HANDLE_COLOR = (40, 40, 40)
	HANDLE_WIDTH = 6
	HANDLE_HEIGHT_OVERHANG = 5


	def __init__(self, x, y, width, height, csg_model, callback):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.callback = callback
		# Display subtract primitives and ignore add primitives.
		self.subtract_only_mode = False
		self.reset(csg_model)


	# Create one segment per primitive and move the handles to both ends of the slider.
	def reset(self, csg_model):
		self.operations = [get_operation_name(command['operation weights']) for command in csg_model.csg_commands]
		self.segment_width = self.width / len(self.operations)
		self.handles = [0, len(self.operations)]
		self.dragged_handle = None


	def draw(self):
		(start, end) = sorted(self.handles)
		title_y = self.y + self.height + self.HANDLE_HEIGHT_OVERHANG + 4
		shapes = [pyglet.text.Label('Primitives', x=self.x - self.HANDLE_WIDTH / 2, y=title_y, anchor_y='bottom', color=(0,0,0), bold=True)]

		for index, operation in enumerate(self.operations):
			if operation == 'sub':
				color = self.SUBTRACT_COLOR
			elif self.subtract_only_mode:
				color = self.HIDDEN_ADD_COLOR
			else:
				color = self.ADD_COLOR

			segment_selected = start <= index < end
			alpha = 255 if segment_selected else self.UNSELECTED_ALPHA
			segment_x = self.x + index * self.segment_width + 1
			shapes.append(pyglet.shapes.Rectangle(segment_x, self.y, self.segment_width - 2, self.height, (*color, alpha)))

		for handle in self.handles:
			handle_x = self.x + handle * self.segment_width
			handle_y = self.y - self.HANDLE_HEIGHT_OVERHANG
			shapes.append(pyglet.shapes.Rectangle(handle_x - self.HANDLE_WIDTH / 2, handle_y, self.HANDLE_WIDTH, self.height + self.HANDLE_HEIGHT_OVERHANG * 2, self.HANDLE_COLOR))
			shapes.append(pyglet.text.Label(str(handle), x=handle_x, y=handle_y - 2, anchor_x='center', anchor_y='top', color=(0,0,0)))

		glDisable(GL_DEPTH_TEST)

		for shape in shapes:
			shape.draw()


	# Grab the handle closest to the mouse and move it under the mouse. Returns True if the slider was clicked.
	def on_mouse_press(self, mouse_x, mouse_y):
		in_slider_width = self.x - self.HANDLE_WIDTH < mouse_x < self.x + self.width + self.HANDLE_WIDTH
		in_slider_height = self.y - self.HANDLE_HEIGHT_OVERHANG < mouse_y < self.y + self.height + self.HANDLE_HEIGHT_OVERHANG

		if not (in_slider_width and in_slider_height):
			return False

		# Grab whichever handle is closer to the mouse
		mouse_position = (mouse_x - self.x) / self.segment_width
		left_handle_dist = abs(self.handles[0] - mouse_position)
		right_handle_dist = abs(self.handles[1] - mouse_position)
		self.dragged_handle = 0 if left_handle_dist <= right_handle_dist else 1
		self.pressed_range = sorted(self.handles)

		return self.on_mouse_drag(mouse_x, mouse_y)


	# Move the dragged handle to the boundary closest to the mouse. Returns True while a handle is being dragged.
	def on_mouse_drag(self, mouse_x, mouse_y):
		if self.dragged_handle is None:
			return False

		dragged_handle = self.handles[self.dragged_handle]
		unselected_handle = self.handles[1 - self.dragged_handle]

		# Move the selected handle to the mouse position.
		new_handle_position = round((mouse_x - self.x) / self.segment_width)
		# Keep the handle within the slider.
		new_handle_position = max(0, min(new_handle_position, len(self.operations)))

		# Handles must be at least one segment apart, so stop next to the other handle until the mouse passes it
		if new_handle_position == unselected_handle:
			if dragged_handle > unselected_handle:
				new_handle_position = unselected_handle + 1
			else:
				new_handle_position = unselected_handle - 1

		self.handles[self.dragged_handle] = new_handle_position
		return True


	# Recompute the displayed model if the selected range changed while the handle was dragged.
	def on_mouse_release(self, mouse_x, mouse_y):
		if self.dragged_handle is not None and sorted(self.handles) != self.pressed_range:
			self.callback(*sorted(self.handles))

		self.dragged_handle = None


class _SdfViewer(pyrender.Viewer):
	LEFT_KEY = 65361
	RIGHT_KEY = 65363
	buttons = []

	def __init__(self, mesh_nodes, window_title, point_size, num_view_points, view_width=1000, view_height=1000):
		self.num_view_points = num_view_points
		self.update_mesh_node()
		self.view_width = view_width
		self.view_height = view_height

		scene = pyrender.Scene(ambient_light=np.array([50,50,50]))

		for mesh_node in mesh_nodes:
			scene.add_node(mesh_node)

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
		raise NotImplementedError("Implement method in inheriting class")


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
		captured = [widget.on_mouse_press(x, y) for widget in self.buttons]

		# Skip camera controls when a widget captures the mouse
		if not any(captured):
			super(_SdfViewer, self).on_mouse_press(x, y, button, modifiers)


	def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
		captured = [widget.on_mouse_drag(x, y) for widget in self.buttons]

		if not any(captured):
			super(_SdfViewer, self).on_mouse_drag(x, y, dx, dy, buttons, modifiers)


	def on_mouse_release(self, x, y, button, modifiers):
		for widget in self.buttons:
			widget.on_mouse_release(x, y)

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
	def __init__(self, window_title, point_size, num_view_points, input_file):
		self.num_view_points = num_view_points
		self.file_loader = FileLoader(input_file, ['*.npy'])
		self.load_file(self.file_loader.get_file())
		self.point_cloud_node = pyrender.Node()

		super(SdfFileViewer, self).__init__(
			[self.point_cloud_node],
			window_title,
			point_size,
			num_view_points)


	def update_mesh_node(self):
		select_points = self.points
		select_distances = self.distances
		colors = torch.zeros(select_points.size())
		colors[select_distances < 0, 2] = 1
		colors[select_distances > 0, 0] = 1
		cloud = pyrender.Mesh.from_points(select_points.cpu().numpy(), colors=colors)
		self.point_cloud_node.mesh = cloud


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

	ALL_OP_MODE = "All OP Mode"
	SUBTRACT_ONLY_MODE = "Subtract Only Mode"


	def __init__(self, window_title, point_size, num_view_points, input_file, sample_dist, get_mesh_and_csg_model=None, recon_resolution=256):
		self.view_mode = self.COMBINED_VIEW
		self.model_mode = self.POINT_MODEL
		self.operation_mode = self.ALL_OP_MODE
		self.num_view_points = num_view_points
		self.recon_resolution = recon_resolution
		self.get_mesh_and_csg_model = get_mesh_and_csg_model
		self.file_loader = FileLoader(input_file, MESH_FILE_TYPES)
		self.sample_dist = sample_dist
		self.load_file(self.file_loader.get_file())

		self.point_cloud_node = pyrender.Node()
		self.original_mesh_node = pyrender.Node()
		self.recon_mesh_node = pyrender.Node()
		mesh_nodes = [self.point_cloud_node, self.original_mesh_node, self.recon_mesh_node]

		super(SdfModelViewer, self).__init__(
			mesh_nodes,
			window_title,
			point_size,
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

		# Create primitive range slider in the top left corner, leaving room for its title
		slider_height = 20
		title_height = 20
		slider_y = self.viewport_size[1] - slider_height - padding - title_height - PrimitiveSlider.HANDLE_HEIGHT_OVERHANG
		self.slider = PrimitiveSlider(padding * 2, slider_y, 400, slider_height, self.full_csg_model, self.set_primitive_range)
		self.add_button(self.slider)

		# Operation Viewing Mode buttons below primitive slider.
		handle_label_height = 20
		op_buttons_width = (self.slider.width - padding) // 2
		op_buttons_y = slider_y - PrimitiveSlider.HANDLE_HEIGHT_OVERHANG - handle_label_height - padding - height
		all_op_button = Button(self.slider.x, op_buttons_y, op_buttons_width, height, 'View All Operations', callback=lambda: self.set_operation_mode(self.ALL_OP_MODE))
		sub_only_button = Button(self.slider.x + op_buttons_width + padding, op_buttons_y, op_buttons_width, height, 'View Subtract Only', callback=lambda: self.set_operation_mode(self.SUBTRACT_ONLY_MODE))
		default_op_button = all_op_button if self.operation_mode is self.ALL_OP_MODE else sub_only_button
		self.add_button(ToggleButtons([all_op_button, sub_only_button], default_op_button))


	# Display only selected primitives. When Subtract Only Mode is enabled, only display subtracted primitives.
	def set_primitive_range(self, start, end):
		self.csg_model = CSGModel(self.full_csg_model.batch_size, self.full_csg_model.device)

		for command in self.full_csg_model.csg_commands[start:end]:
			# Copy over model outputs to CSG object normally.
			if self.operation_mode is self.ALL_OP_MODE:
				self.csg_model.add_command_dict(command)
			# When in Subtract Only mode, only display subtract operations.
			elif get_operation_name(command['operation weights']) == 'sub':
				# Flip add and subtract operations.
				add_weights = command['operation weights'].flip(-1)
				self.csg_model.add_command_dict({**command, 'operation weights': add_weights, 'blending': None})

		# Display a placeholder sphere when there are no primitives to display
		if self.csg_model.num_commands == 0:
			self.csg_model.add_command_dict(dict(self.full_csg_model.csg_commands[0]))
			self.csg_model.replace_empty_shapes(torch.ones(self.csg_model.batch_size, dtype=torch.bool, device=self.csg_model.device))

		self.recon_mesh = csg_to_mesh(self.csg_model, self.recon_resolution)[0]
		self.recon_surface_points = sample_points_mesh_surface(self.recon_mesh, self.num_view_points)
		self.update_mesh_node()


	def set_operation_mode(self, mode):
		self.operation_mode = mode
		self.slider.subtract_only_mode = mode is self.SUBTRACT_ONLY_MODE
		self.set_primitive_range(*sorted(self.slider.handles))


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
				color = np.array([0, 0, 255])
				self.view_point_cloud(self.target_surface_points, color)
				return

			case self.COMBINED_VIEW:
				self.view_combined_point_cloud()
				return

			case self.RECON_VIEW:
				color = np.array([255, 0, 0])
				self.view_point_cloud(self.recon_surface_points, color)
				return


	def update_mesh(self):
		display_original = True
		display_recon = True
		alpha = 255

		match self.view_mode:
			case self.ORIGINAL_VIEW:
				display_recon = False

			case self.COMBINED_VIEW:
				alpha = 0.5

			case self.RECON_VIEW:
				display_original = False

		blueMaterial = pyrender.MetallicRoughnessMaterial(baseColorFactor=np.array([0,0,255, alpha]), alphaMode="BLEND")
		redMaterial = pyrender.MetallicRoughnessMaterial(baseColorFactor=np.array([255,0,0, alpha]), alphaMode="BLEND")
		self.original_mesh_node.mesh = pyrender.Mesh.from_trimesh(self.target_mesh, material=blueMaterial)
		self.recon_mesh_node.mesh = pyrender.Mesh.from_trimesh(self.recon_mesh, material=redMaterial)
		self.original_mesh_node.mesh.is_visible = display_original
		self.recon_mesh_node.mesh.is_visible = display_recon


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
		self.original_mesh_node.mesh = cloud
		self.recon_mesh_node.mesh = cloud


	def view_point_cloud(self, surface_points, colors):
		# Format colors array
		colors = np.expand_dims(colors, axis=0).repeat(surface_points.shape[0], axis=0)

		# Sample reconstructed CSG model
		cloud = pyrender.Mesh.from_points(surface_points.squeeze(0).cpu().numpy(), colors=colors)
		self.original_mesh_node.mesh = cloud
		self.recon_mesh_node.mesh = cloud


	def load_file(self, samples_file):
		self.target_mesh, self.recon_mesh, self.full_csg_model = self.get_mesh_and_csg_model(samples_file)
		self.csg_model = self.full_csg_model
		self.target_surface_points = sample_points_mesh_surface(self.target_mesh, self.num_view_points)
		self.recon_surface_points = sample_points_mesh_surface(self.recon_mesh, self.num_view_points)


	def view_prev(self):
		if self.get_mesh_and_csg_model == None:
			return

		samples_file = self.file_loader.prev_file()
		self.load_file(samples_file)
		self.slider.reset(self.full_csg_model)
		self.set_primitive_range(*sorted(self.slider.handles))


	def view_next(self):
		if self.get_mesh_and_csg_model == None:
			return

		samples_file = self.file_loader.next_file()
		self.load_file(samples_file)
		self.slider.reset(self.full_csg_model)
		self.set_primitive_range(*sorted(self.slider.handles))


# Parse commandline arguments
def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', required=True, type=str, help='Directory or Numpy file of sample points and distance values.')
	parser.add_argument('--num_view_points', type=int, default=-1, help='Number of points to display. Set to -1 to display all points.')
	parser.add_argument('--point_size', type=int, default=3, help='Size to render each point of the point cloud.')

	args = parser.parse_args()
	return args


def main():
	args = options()
	print('')

	try:
		viewer = SdfFileViewer("View SDF", args.point_size, args.num_view_points, args.input_file)
	except FileNotFoundError as fileError:
		print(fileError)
	except Exception:
		print(traceback.format_exc())



if __name__ == '__main__':
	main()