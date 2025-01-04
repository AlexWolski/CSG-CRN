import os
import math
import torch
import trimesh
import tkinter as tk
from tkinter import filedialog, simpledialog, StringVar
from pytorch3d.ops.marching_cubes import marching_cubes
from utilities.csg_model import MIN_BOUND, MAX_BOUND


# Estimated multiplicative memory usage factor to compute an SDF for any given input size
SDF_MEMORY_USAGE_FACTOR = 6


def get_grid_points(min_bound, max_bound, resolution, device=None):
	"""
	Generates a tuple of there tensors with size (N,N,N). The three tensors represent X, Y, and Z axis values respectively.
	Each point represents a voxel position in a voxel grid. Note that the tensors are non-contiguous to save memory.

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
	Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
		Three tensors of size (N, N, N) where N=`resolution`.
		Each tensor represents an axis of a voxel grid.

	"""
	values = torch.linspace(min_bound, max_bound, resolution, device=device)
	x = values.expand([resolution, resolution, resolution])
	y = x.transpose(1,2)
	z = x.transpose(0,2)
	return (x, y, z)


def csg_to_mesh(csg_model, resolution, iso_level=0.0):
	torch.cuda.empty_cache()

	with torch.no_grad():
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
		# Generate grid points
		voxel_size = abs(MAX_BOUND - MIN_BOUND) / resolution
		(x, y, z) = get_grid_points(MIN_BOUND, MAX_BOUND, resolution, csg_model.device)

		# Calculate number of batches needed to compute SDF values given memory restrictions
		(free_memory, total_memory) = torch.cuda.mem_get_info(device=x.device.index)
		num_samples = resolution**3
		input_bytes = num_samples * 3 * x.element_size()
		output_bytes = num_samples * x.element_size()
		comp_bytes = input_bytes * SDF_MEMORY_USAGE_FACTOR
		total_req_bytes = input_bytes + comp_bytes
		num_batches = math.ceil(total_req_bytes / (free_memory - output_bytes))

		# Quantize batch size to be divisible by grid layer size
		layers_per_batch = resolution // num_batches

		distances_list = []

		# Sample SDF
		for layer in range(0, resolution, layers_per_batch):
			start_layer = layer
			end_layer = layer + layers_per_batch if layer + layers_per_batch < resolution else resolution

			# Slice query points in current batch
			x_slice = x[start_layer:end_layer,...]
			y_slice = y[start_layer:end_layer,...]
			z_slice = z[start_layer:end_layer,...]
			# Combine query points into single tensor
			points_slice = torch.stack((x_slice, y_slice, z_slice), dim=-1)
			# Reshape query points to (1, N, 3) where N=num_points
			points_slice = points_slice.reshape(1,-1,3)
			# Sample SDF
			distances_list.append(csg_model.sample_csg(points_slice))

		# Reshape into grid
		distances = torch.cat(distances_list, dim=-1)
		distances = distances.reshape(1, resolution, resolution, resolution)

		del x, y ,z, distances_list

		try:
			verts, faces = marching_cubes(distances, isolevel=0.0, return_local_coords=True)
			mesh = trimesh.Trimesh(verts[0].cpu(), faces[0].cpu())
		except torch.OutOfMemoryError:
			verts, faces = marching_cubes(distances.cpu(), isolevel=0.0, return_local_coords=True)
			mesh = trimesh.Trimesh(verts[0].cpu(), faces[0].cpu())
		finally:
			torch.cuda.empty_cache()
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
	export_dialog.title('Input Dialog Example')

	res_label = tk.Label(export_dialog, text='Resolution:')
	res_label.grid(column=0, row=0, padx=(20,0), pady=(20,0))

	options = [
		'64',
		'128',
		'256',
		'512',
		'1024',
	]

	resolution_string = StringVar()
	resolution_string.set('256')
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

	export_button = tk.Button(export_dialog, text='Export', command=export_mesh)
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

	try:
		export_to_mesh(csg_model, resolution, output_file)
	except torch.OutOfMemoryError as e:
		print('Insufficient memory on target device. Try selecting a lower resolution or changing compute device.')
		print(e)
	except Exception as e:
		print('Unexpected Error:')
		print(e)
	finally:
		torch.cuda.empty_cache()
