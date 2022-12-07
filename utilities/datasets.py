import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PointDataset(Dataset):
	def __init__(self, uniform_dir, surface_dir, filenames, num_points, device):
		self.uniform_dir = uniform_dir
		self.surface_dir = surface_dir
		self.filenames = filenames
		self.num_points = num_points
		self.device = device

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		filename = self.filenames[idx]

		# Load uniform points from file
		uniform_path = os.path.join(self.uniform_dir, filename)
		uniform_points = np.load(uniform_path)

		# Load surface points from file
		surface_path = os.path.join(self.surface_dir, filename)
		surface_points = np.load(surface_path)

		# Randomly select needed number of surface points
		replace = (surface_points.shape[0] < self.num_points)
		select_rows = np.random.choice(surface_points.shape[0], self.num_points, replace=replace)
		surface_points = surface_points[select_rows]

		# Convert numpy arrays to torch tensors
		uniform_points = torch.from_numpy(uniform_points)
		surface_points = torch.from_numpy(surface_points)

		return (uniform_points, surface_points)