import os
import numpy as np
from torch.utils.data import Dataset


class PointDataset(Dataset):
	def __init__(self, data_dir, filenames, num_points):
		self.data_dir = data_dir
		self.filenames = filenames
		self.num_points = num_points

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		# Load all points from file
		filename = self.filenames[idx]
		points_path = os.path.join(self.data_dir, filename)
		points = np.load(points_path)

		# Randomly select needed number of points
		replace = (points.shape[0] < self.num_points)
		select_rows = np.random.choice(points.shape[0], self.num_points, replace=replace)
		select_points = points[select_rows]

		return select_points