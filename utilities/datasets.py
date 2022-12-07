import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PointDataset(Dataset):
	def __init__(self, data_dir, filenames, num_points, device):
		self.data_dir = data_dir
		self.filenames = filenames
		self.num_points = num_points
		self.device = device

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		# Load all points from file
		filename = self.filenames[idx]
		points_path = os.path.join(self.data_dir, filename)
		points = np.load(points_path)

		# Randomly select needed number of surface points
		replace = (points.shape[0] < self.num_points)
		select_rows = np.random.choice(points.shape[0], self.num_points, replace=replace)
		select_points = points[select_rows]

		# Convert numpy arrays to torch tensors
		select_points = torch.from_numpy(points)

		return (points, select_points)