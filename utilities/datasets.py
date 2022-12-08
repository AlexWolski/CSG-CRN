import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PointDataset(Dataset):
	def __init__(self, data_dir, file_rel_paths, num_input_points, num_loss_points):
		self.data_dir = data_dir
		self.file_rel_paths = file_rel_paths
		self.num_input_points = num_input_points
		self.num_loss_points = num_loss_points

	def __len__(self):
		return len(self.file_rel_paths)

	def __getitem__(self, idx):
		# Load all points from file
		file_rel_path = self.file_rel_paths[idx]
		points_path = os.path.join(self.data_dir, file_rel_path)
		points = np.load(points_path)

		# Randomly select needed number of input surface points
		replace = (points.shape[0] < self.num_input_points)
		select_rows = np.random.choice(points.shape[0], self.num_input_points, replace=replace)
		select_input_points = points[select_rows]

		# Randomly select needed number of loss surface points
		replace = (points.shape[0] < self.num_loss_points)
		select_rows = np.random.choice(points.shape[0], self.num_input_points, replace=replace)
		select_loss_points = points[select_rows]

		# Convert numpy arrays to torch tensors
		select_input_points = torch.from_numpy(select_input_points)
		select_loss_points = torch.from_numpy(select_loss_points)

		return (select_input_points, select_loss_points)