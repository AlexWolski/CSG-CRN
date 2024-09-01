import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utilities.data_augmentation import augment_sample


class PointDataset(Dataset):
	def __init__(self, file_rel_paths, args):
		self.file_rel_paths = file_rel_paths
		self.raw_copies = len(file_rel_paths)
		self.augmented_copies = len(file_rel_paths) * args.augment_copies
		self.args = args

	def __len__(self):
		return self.augmented_copies

	def __getitem__(self, idx):
		# Load all points and distances from sdf sample file
		index = idx % self.raw_copies
		file_rel_path = self.file_rel_paths[index]
		sample_path = os.path.join(self.args.data_dir, file_rel_path)
		sdf_sample = np.load(sample_path).astype(np.float32)
		sdf_sample = torch.from_numpy(sdf_sample)

		# Augment sample
		if self.args.augment_data:
			points = sdf_sample[:,:3]
			distances = sdf_sample[:,3]
			distances = distances.unsqueeze(0).transpose(0, 1)

			augmented_points, augmented_distances = augment_sample(points, distances, self.args)

			sdf_sample = torch.cat((augmented_points, augmented_distances), dim=1)

		# Randomly select indices for needed number of input surface samples
		replace = (sdf_sample.shape[0] < self.args.num_input_points)
		select_input_rows = np.random.choice(sdf_sample.shape[0], self.args.num_input_points, replace=replace)

		# Randomly select indices for needed number of loss surface samples
		replace = (sdf_sample.shape[0] < self.args.num_loss_points)
		select_loss_rows = np.random.choice(sdf_sample.shape[0], self.args.num_loss_points, replace=replace)

		# Index the sdf tensor to get input and loss tensors
		select_input_samples = sdf_sample[select_input_rows]
		select_loss_samples = sdf_sample[select_loss_rows]

		return (select_input_samples, select_loss_samples)