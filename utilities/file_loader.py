import os
import glob
from pathlib import Path


class FileLoader:
	def __init__(self, input_file, file_type_list):
		self.file_list = []

		input_file_path = Path(input_file)

		if os.path.isdir(input_file):
			self.parent_dir = input_file_path.absolute()
			self.file_index = 0
			input_file_name = None
		elif os.path.isfile(input_file):
			self.parent_dir = input_file_path.parent.absolute()
			input_file_name = input_file_path.name
		else:
			raise FileNotFoundError(f'Provided input file is not a valid file or directory:\n{input_file}')

		# Find all valid files in parent directory
		for mesh_file_type in file_type_list:
			self.file_list.extend(glob.glob(os.path.join(self.parent_dir, '**', mesh_file_type), recursive=True))

		# TODO: File file index
		self.file_index = 0

		if len(self.file_list) == 0:
			raise FileNotFoundError(f'No model files found in the directory. Valid model file types are:\n{file_type_list}')


	def get_file(self):
		file_name = self.file_list[self.file_index]
		file_path = self.parent_dir.joinpath(file_name)
		return file_path


	def prev_file(self):
		self.file_index -= 1

		if self.file_index < 0:
			self.file_index = len(self.file_list) - 1

		return self.get_file()


	def next_file(self):
		self.file_index += 1

		if self.file_index >= len(self.file_list):
			self.file_index = 0

		return self.get_file()
