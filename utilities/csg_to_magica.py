import copy
import json
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation
from utilities.csg_model import CSGModel, get_primitive_name, get_operation_name


# Amount to scale the output 
SCALE_FACTOR = 100
# Scalar multiplier when converting the blending strength to MagicaCSG
BLENDING_SCALAR = 100
# Marching cubes resolution in MagicaCSG
RESOLUTION = 128

# Template json format for MagicaCSG
magica_dict = {
	'meta': {
		'version': '2',
	},
	'object':
	[
		{
			'type': 'csg',
			'res': f'{RESOLUTION}',
			't': '0 0 0',
		},
	],
	'csg': [[]],
}


# Convert shape weight tensor to MagicaCSG shape string
def export_primitive(shape_weights):
	return get_primitive_name(shape_weights)


# Convert operation weight tensor to MagicaCSG mode string
def export_operation(operation_weights):
	return get_operation_name(operation_weights)


# Adjust the blending value to match the MagicaCSG 
def export_blending(blending):
	blending_value = blending.item()
	blending_value *= BLENDING_SCALAR
	return str(blending_value)


# Convert roundness value to string
def export_roundness(roundness):
	return str(roundness.item())


# Scale and export the shape position
def export_position(position):
	position = position.squeeze(0) * SCALE_FACTOR
	position_list = position.tolist()
	position_string = ' '.join(str(pos) for pos in position_list)
	return position_string


# Convert quaternion tensor to MagicaCSG rotation matrix string
def export_quaternion(quaternion):
	quaternion_list = quaternion.squeeze(0).tolist()
	rotation_matrix = Rotation.from_quat(quaternion_list, scalar_first=True).as_matrix()
	rotation_string = ''

	# Convert the matrix to a string
	for col in range(rotation_matrix.shape[1]):
		for row in range(rotation_matrix.shape[0]):
			rotation_string += str(rotation_matrix[row][col]) + ' '

	return rotation_string


# Scale and export the shape dimensions
def export_scale(scale):
	scale = scale.squeeze(0) * SCALE_FACTOR
	scale_list = scale.tolist()
	scale_string = ' '.join(str(scale) for scale in scale_list)
	return scale_string


# Convert CSGModel command to MagicaCSG dict
def command_to_json(command):
	command_dict = {}
	command_dict['type'] = export_primitive(command['shape weights'])
	command_dict['mode'] = export_operation(command['operation weights'])
	command_dict['t'] = export_position(command['translations'])
	command_dict['r'] = export_quaternion(command['rotations'])
	command_dict['s'] = export_scale(command['scales'])

	if command['blending'] is not None:
		command_dict['blend'] = export_blending(command['blending'])

	if command['roundness'] is not None:
		command_dict['round%'] = export_roundness(command['roundness'])

	return command_dict


# Convert CSGModel object to MagicaCSG format string
def csg_to_magica(csg_model):
	output_dict = copy.deepcopy(magica_dict)
	output_string = ''

	# Add each CSG command to the output dict
	for command in csg_model.csg_commands:
		output_dict['csg'][0].append(command_to_json(command))

	# Format each key/value pair in the dict as a json
	for key in output_dict:
		value = json.dumps(output_dict[key], indent=2, separators=('', ':'))
		output_string += f'{key}: {value}\n'

	return output_string


# Convert CSG model to MagicaCSG format and save to file prompted from user
def prompt_and_export_to_magica(csg_model):
	output_string = csg_to_magica(csg_model)
	output_file = filedialog.asksaveasfile(filetypes=[('Magica CSG', '*.mcsg')])

	if output_file:
		output_file.write(output_string)
		output_file.close()
