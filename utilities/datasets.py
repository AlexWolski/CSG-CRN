import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PointDataset(Dataset):
	def __init__(self, file_rel_paths, args):
		self.file_rel_paths = file_rel_paths
		self.args = args

	def __len__(self):
		return len(self.file_rel_paths)

	def __getitem__(self, idx):
		# Load all points and distances from sample file
		file_rel_path = self.file_rel_paths[idx]
		sample_path = os.path.join(self.args.data_dir, file_rel_path)
		sample = np.load(sample_path).astype(np.float32)

		# Randomly select needed number of input surface samples
		replace = (sample.shape[0] < self.args.num_input_points)
		select_rows = np.random.choice(sample.shape[0], self.args.num_input_points, replace=replace)
		select_input_samples = sample[select_rows]

		# Randomly select needed number of loss surface samples
		replace = (sample.shape[0] < self.args.num_loss_points)
		select_rows = np.random.choice(sample.shape[0], self.args.num_loss_points, replace=replace)
		select_loss_samples = sample[select_rows]

		# Convert numpy arrays to torch tensors
		select_input_samples = torch.from_numpy(select_input_samples)
		select_loss_samples = torch.from_numpy(select_loss_samples)

		return (select_input_samples, select_loss_samples)