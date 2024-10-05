import os
from pathlib import Path

class FileLoader:
	def __init__(self, input_file):
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


	def prev_file(self):
		self.file_index -= 1

		if self.file_index < 0:
			self.file_index = len(self.file_list) - 1;

		file_name = self.file_list[self.file_index]
		file_path = self.parent_dir.joinpath(file_name)
		return file_path


	def next_file(self):
		self.file_index += 1

		if self.file_index >= len(self.file_list):
			self.file_index = 0;

		file_name = self.file_list[self.file_index]
		file_path = self.parent_dir.joinpath(file_name)
		return file_path