import os
import torch
import trimesh
import tkinter as tk
from tkinter import filedialog, simpledialog, StringVar
from skimage import measure
from utilities.csg_model import MIN_BOUND, MAX_BOUND


def get_grid_points(min_bound, max_bound, resolution, device=None):
	"""
	Generates a (N,N,N,3) grid of points where each point represents the position of a voxel.

	Parameters
	----------
	min_bound : float
		Minimum value along each axis of the grid.
	max_bound : float
		Maximum value along each axis of the grid.
	resolution : int
		Number of voxels to create along each axis. Grid contains `resolution`^3 total voxels.
	device : string, optional
		String representing device to create new grid tensor on. Default is None.

	Returns
	-------
	torch.Tensor
		Tensor of size (N, N, N, 3) where N=`resolution`.
		Represents a voxel grid where each of the N^3 points represents the position of a voxel.

	"""
	axis_values = torch.linspace(min_bound, max_bound, resolution, device=device)
	grid_axes = torch.meshgrid(axis_values, axis_values, axis_values, indexing='ij')
	grid_points = torch.stack(grid_axes, dim=-1)
	return grid_points.detach()


def csg_to_mesh(csg_model, resolution, iso_level=0.0):
	"""
	Use the marching cubes algorithm to extract a mesh representation of a given implicit CSG model.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		CSG model object to convert to a mesh.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	iso_level : float, optional
		The distance from the  isosurface. Default is 0.

	Returns
	-------
	trimesh.Trimesh
		Converted mesh object.

	"""
	voxel_size = abs(MAX_BOUND - MIN_BOUND) / resolution
	grid_points = get_grid_points(MIN_BOUND, MAX_BOUND, resolution, csg_model.device)

	# Reshape to (B, N, 3) where B=batch_size and N=num_points
	flat_points = grid_points.reshape(1, -1, 3)
	flat_distances = csg_model.sample_csg(flat_points)
	grid_distances = flat_distances.reshape(resolution, resolution, resolution)

	# Send distances tensor to CPU and convert to numpy
	grid_distances = grid_distances.cpu().detach().numpy()
	verts, faces, normals, values = measure.marching_cubes(grid_distances, level=0.0, spacing=[voxel_size] * 3)

	# Transform vertices from voxel space to object space
	offset = abs(MAX_BOUND - MIN_BOUND) / 2.0
	verts = verts - offset

	# Generate mesh
	mesh = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
	return mesh


def sample_csg_surface(csg_model, resolution, num_samples):
	"""
	Uniformly sample points on the surface of an implicit CSG model.
	Uses the marching cubes algorithm to extract an isosurface mesh, then uniformly samples the mesh faces.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	num_samples: int
		Number of surface samples to generate.

	Returns
	-------
	torch.Tensor
		Tensor of size (N, 3) where N=`num_samples`.
		Each point in the tensor is approximately on the surface of the given CSG model.

	"""
	mesh = csg_to_mesh(csg_model, resolution)
	samples = trimesh.sample_surface(mesh, num_samples)
	return torch.from_numpy(samples).detach()


def export_to_mesh(csg_model, resolution, output_file):
	"""
	Use the marching cubes algorithm to extract a mesh from a CSG Model and save the result to file.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.
	output_file: string
		Path to save the exported mesh.

	"""
	with torch.no_grad():
		mesh = csg_to_mesh(csg_model, resolution)

	file_extention = output_file.split('.')[-1]
	trimesh.exchange.export.export_mesh(mesh, output_file, file_extention)


def prompt_export_settings():
	"""
	Display a GUI to prompt the user for a marching cubes resolution and output file.

	Returns
	-------
	file_path : string
		Path to mesh output file.
	resolution : int
		Marching cubes resolution.

	"""
	export_dialog = tk.Tk()
	export_dialog.title("Input Dialog Example")

	res_label = tk.Label(export_dialog, text="Resolution:")
	res_label.grid(column=0, row=0, padx=(20,0), pady=(20,0))

	options = [
		"64",
		"128",
		"256",
		"512",
	]

	resolution_string = StringVar()
	resolution_string.set( "256" )
	file_path = StringVar()

	res_option = tk.OptionMenu(export_dialog, resolution_string, *options)
	res_option.grid(column=1, row=0, padx=(0,10), pady=(10,0))

	filetypes = [
		('GLB File', '.glb'),
		('GLTF File', '.gltf'),
		('OBJ File', '.obj'),
		('OFF File', '.off'),
		('PLY File', '.ply'),
		('STL File', '.stl'),
	]

	export_mesh = lambda _=export_dialog: (
		file_path.set(filedialog.asksaveasfilename(filetypes=filetypes)),
		export_dialog.destroy(),
	)

	export_button = tk.Button(export_dialog, text="Export", command=export_mesh)
	export_button.grid(column=0, row=2, columnspan=2, pady=(30, 10))

	export_dialog.mainloop()
	return (file_path.get(), int(resolution_string.get()))


def prompt_and_export_to_mesh(csg_model):
	"""
	Use the marching cubes algorithm to extract a mesh from the given CSG Model save it to user-specified file.

	Parameters
	----------
	csg_model : utilities.csg_model.CSGModel
		The CSG model to sample.
	resolution : int
		Voxel resolution to use for the marching cubes algorithm.

	"""
	(output_file, resolution) = prompt_export_settings()

	if not output_file:
		return

	export_to_mesh(csg_model, resolution, output_file)
