import os
import math
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utilities.data_augmentation import augment_sample_batch
from utilities.data_processing import UNIFORM_FOLDER, SURFACE_FOLDER, NEAR_SURFACE_FOLDER


class PointDataset(Dataset):
	def __init__(self, file_rel_paths, device, args, dataset_name="Dataset"):
		self.file_rel_paths = file_rel_paths
		self.dataset_name = dataset_name
		self.raw_copies = len(file_rel_paths)
		self.device = device
		self.augmented_copies = len(file_rel_paths) * args.augment_copies
		self.args = args
		self.__load_data_set();


	def __load_data_set(self):
		data_sample_list = []
		skipped_samples = 0

		# Load all samples
		for file_rel_path in tqdm(self.file_rel_paths, desc=f'Loading {self.dataset_name}'):
			data_sample = self.__load_data_sample(file_rel_path)

			# Skip samples that fail to load
			if data_sample == None:
				skipped_samples += 1
				continue

			data_sample_list.append(data_sample)

		# Omit skipped samples
		self.raw_copies = len(self.file_rel_paths) - skipped_samples
		# Convert sample list to tensor and save in system memory
		self.sdf_samples = torch.stack(data_sample_list, dim=0)


	def __load_data_sample(self, file_rel_path):
		# Compute number of uniform and near-surface SDF samples to load
		total_sdf_samples = self.args.num_input_points + self.args.num_loss_points
		num_uniform_samples = math.ceil(total_sdf_samples * self.args.surface_uniform_ratio)
		num_surface_samples = math.floor(total_sdf_samples * (1 - self.args.surface_uniform_ratio))

		# Load uniform and near-surface SDF samples from file
		uniform_samples = self.__load_point_samples(UNIFORM_FOLDER, file_rel_path, num_uniform_samples)
		near_surface_samples = self.__load_point_samples(NEAR_SURFACE_FOLDER, file_rel_path, num_surface_samples)

		# Skip samples that don't load correctly
		if uniform_samples == None or near_surface_samples == None:
			return None

		# Combine all SDF samples into one tensor
		return torch.cat((uniform_samples, near_surface_samples), dim=0)


	# Load a specified number of point samples from a numpy file
	def __load_point_samples(self, subfolder, file_rel_path, num_point_samples=None):
		# Open numpy file as memmap
		file_path = os.path.join(self.args.data_dir, subfolder, file_rel_path)
		samples_mmap = np.load(file_path, mmap_mode='r')

		if num_point_samples == None:
			num_point_samples = samples_mmap.shape[0]
		elif num_point_samples > samples_mmap.shape[0]:
			return None

		# Copy the required number of samples into memory and convert to a torch tensor
		samples = samples_mmap[:num_point_samples].copy().astype(np.float32)
		return torch.from_numpy(samples)


	def __len__(self):
		return self.augmented_copies

	def __getitem__(self, batch_idx):
		# Adjust indices for augmented copies
		if self.augmented_copies > self.raw_copies:
			batch_idx = [index % self.raw_copies for index in batch_idx]

		batch_sdf_samples = self.sdf_samples[batch_idx].to(self.device)

		# Augment samples
		if self.args.augment_data:
			batch_sdf_points = batch_sdf_samples[:,:,:3]
			batch_sdf_distances = batch_sdf_samples[:,:,3]
			batch_sdf_distances = batch_sdf_distances.unsqueeze(2)

			augmented_points, augmented_distances = augment_sample_batch(batch_sdf_points, batch_sdf_distances, self.args)
			batch_sdf_samples = torch.cat((augmented_points, augmented_distances), dim=2)

		# Shuffle the data samples
		total_points = self.args.num_input_points + self.args.num_loss_points
		batch_sdf_samples = batch_sdf_samples[:, torch.randperm(total_points)]

		# Separate input and loss samples
		batch_target_input_samples = batch_sdf_samples[:,:self.args.num_input_points].detach()
		batch_target_loss_samples = batch_sdf_samples[:,self.args.num_input_points:].detach()

		# TODO: Implement initial recon loading
		batch_recon_input_samples = None
		batch_recon_loss_samples = None

		return (batch_target_input_samples, batch_target_loss_samples, batch_recon_input_samples, batch_recon_loss_samples)
