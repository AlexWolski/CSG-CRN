import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utilities.data_augmentation import augment_sample


class PointDataset(Dataset):
	def __init__(self, file_rel_paths, device, args):
		self.file_rel_paths = file_rel_paths
		self.raw_copies = len(file_rel_paths)
		self.device = device
		self.augmented_copies = len(file_rel_paths) * args.augment_copies
		self.args = args

	def __len__(self):
		return self.augmented_copies

	def __getitem__(self, idx):
		# Load all points and distances from sdf sample file
		index = idx % self.raw_copies
		total_points = self.args.num_input_points + self.args.num_loss_points
		file_rel_path = self.file_rel_paths[index]
		sample_path = os.path.join(self.args.data_dir, file_rel_path)
		sdf_sample = np.load(sample_path).astype(np.float32)
		sdf_sample = torch.from_numpy(sdf_sample)

		# Send all data to training device
		sdf_sample = sdf_sample.to(self.device)

		# Shuffle the data samples
		sdf_sample = sdf_sample[torch.randperm(total_points)]

		# Augment samples
		if self.args.augment_data:
			points = sdf_sample[:,:3]
			distances = sdf_sample[:,3]
			distances = distances.unsqueeze(0).transpose(0, 1)

			augmented_points, augmented_distances = augment_sample(points, distances, self.args)
			sdf_sample = torch.cat((augmented_points, augmented_distances), dim=1)

		# Separate input and loss samples
		select_input_samples = sdf_sample[:self.args.num_input_points]
		select_loss_samples = sdf_sample[self.args.num_input_points:]

		return (select_input_samples, select_loss_samples)