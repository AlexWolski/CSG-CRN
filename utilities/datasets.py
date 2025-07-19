import os
import math
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from torch.utils.data import Dataset
from losses.loss import Loss
from utilities.data_augmentation import augment_sample_batch, augment_sample_batch_points
from utilities.data_processing import UNIFORM_FOLDER, SURFACE_FOLDER, NEAR_SURFACE_FOLDER

from multiprocessing import Pool


class PointDataset(Dataset):
	# Number of uniform points to load for each required near-surface point
	NEAR_SURFACE_SAMPLE_FACTOR = 10


	def __init__(self, file_rel_paths, device, args, augment_data=False, dataset_name="Dataset", sampling_method=Loss.UNIFIED_SAMPLING):
		self.file_rel_paths = file_rel_paths
		self.augmented_copies = len(file_rel_paths) * args.augment_copies
		self.raw_copies = len(file_rel_paths)
		self.device = device
		self.args = args
		self.augment_data = augment_data
		self.dataset_name = dataset_name
		self.sampling_method = sampling_method


		# Compute number of uniform and near-surface SDF samples to load
		total_sdf_samples = self.args.num_input_points + self.args.num_loss_points

		self.num_uniform_input_samples = math.ceil(self.args.num_input_points * self.args.surface_uniform_ratio)
		self.num_near_surface_input_samples = self.args.num_input_points - self.num_uniform_input_samples
		self.num_uniform_loss_samples = math.ceil(self.args.num_loss_points * self.args.surface_uniform_ratio)
		self.num_near_surface_loss_samples = self.args.num_loss_points - self.num_uniform_loss_samples

		# When the loss is computed on both target and reconstruction near-surface samples,
		# increase the number of uniform points loaded to allow proximity selection.
		if self.sampling_method == Loss.UNIFIED_SAMPLING:
			num_target_loss_samples = math.ceil(self.num_near_surface_loss_samples * Loss.TARGET_RECON_SAMPLING_RATIO)
			num_recon_loss_samples = self.num_near_surface_loss_samples - num_target_loss_samples
			self.num_near_surface_loss_samples = num_target_loss_samples + (num_recon_loss_samples * self.NEAR_SURFACE_SAMPLE_FACTOR)

		self.num_uniform_samples = self.num_uniform_input_samples + self.num_uniform_loss_samples
		self.num_near_surface_samples = self.num_near_surface_input_samples + self.num_near_surface_loss_samples

		self.__load_data_set()


	# Load all samples into system memory
	def __load_data_set(self):
		data_sample_list = []

		for file_rel_path in tqdm(self.file_rel_paths, desc=f'Loading {self.dataset_name}'):
			data_sample_list.append(self.__load_data_sample(file_rel_path))

		# Omit skipped samples
		skipped_samples = data_sample_list.count(None)
		self.raw_copies = len(self.file_rel_paths) - skipped_samples

		if skipped_samples > 0:
			data_sample_list = [i for i in data_sample_list if i != None]

		# Save sample lists as tensors
		uniform_sample_list, near_surface_sample_list, surface_sample_list = list(zip(*data_sample_list))
		self.uniform_samples = torch.stack(uniform_sample_list, dim=0)
		self.near_surface_samples = torch.stack(near_surface_sample_list, dim=0)
		self.surface_samples = torch.stack(surface_sample_list, dim=0)


	def __load_data_sample(self, file_rel_path):
		try:
			# Load SDF samples
			(uniform_samples, near_surface_samples) = self.__load_sdf_samples(file_rel_path)
			# Load surface samples
			surface_samples = self.__load_surface_samples(file_rel_path)

			return (uniform_samples, near_surface_samples, surface_samples)

		# Skip samples that fail to load
		except Exception as e:
			print(e)
			return None


	def __load_sdf_samples(self, file_rel_path):
		# Load uniform and near-surface SDF samples from file
		uniform_samples = self.__load_point_samples(UNIFORM_FOLDER, file_rel_path, self.num_uniform_samples)
		near_surface_samples = self.__load_point_samples(NEAR_SURFACE_FOLDER, file_rel_path, self.num_near_surface_samples)

		return (uniform_samples, near_surface_samples)


	def __load_surface_samples(self, file_rel_path):
		return self.__load_point_samples(SURFACE_FOLDER, file_rel_path, self.args.num_val_acc_points)


	# Load a specified number of point samples from a numpy file
	def __load_point_samples(self, subfolder, file_rel_path, num_point_samples=None):
		# Open numpy file as memmap
		file_path = os.path.join(self.args.data_dir, subfolder, file_rel_path)

		if not os.path.isfile(file_path):
			print('FILE')
			FileNotFoundError(f'Unable to find data sample file:\n{file_path}')

		samples_mmap = np.load(file_path, mmap_mode='r')
		file_length = samples_mmap.shape[0]

		if num_point_samples == None:
			num_point_samples = file_length
		elif num_point_samples > file_length:
			print('LENGTH')
			raise Exception(f'Failed to read {num_point_samples} samples from file with {file_length} samples:\n{file_path}')

		# Copy the required number of samples into memory and convert to a torch tensor
		samples = samples_mmap[:num_point_samples].copy().astype(np.float32)
		return torch.from_numpy(samples)


	def __augment_sdf_samples(self, sdf_samples):
		sdf_points = sdf_samples[:,:,:3]
		sdf_distances = sdf_samples[:,:,3]
		sdf_distances = sdf_distances.unsqueeze(-1)

		augmented_points, augmented_distances = augment_sample_batch(sdf_points, sdf_distances, self.args)
		return torch.cat((augmented_points, augmented_distances), dim=-1)


	def __len__(self):
		return self.augmented_copies


	def __getitem__(self, batch_idx):
		# Adjust indices for augmented copies
		if self.augmented_copies > self.raw_copies:
			batch_idx = [index % self.raw_copies for index in batch_idx]

		# Load batch samples and send to target device
		batch_uniform_samples = self.uniform_samples[batch_idx].to(self.device)
		batch_near_surface_samples = self.near_surface_samples[batch_idx].to(self.device)
		batch_surface_samples = self.surface_samples[batch_idx].to(self.device) if self.surface_samples != None else None

		# Augment samples
		if self.augment_data:
			combined_samples = torch.cat((batch_uniform_samples, batch_near_surface_samples), dim=1)
			combined_samples = self.__augment_sdf_samples(combined_samples)
			batch_uniform_samples = combined_samples[:, self.num_uniform_samples:]
			batch_near_surface_samples = combined_samples[:, :self.num_near_surface_samples]

		# Separate input and loss samples
		batch_uniform_input_samples = batch_uniform_samples[:, :self.num_uniform_input_samples]
		batch_uniform_loss_samples = batch_uniform_samples[:, self.num_uniform_loss_samples:]
		batch_near_surface_input_samples = batch_near_surface_samples[:, :self.num_near_surface_input_samples]
		batch_near_surface_loss_samples = batch_near_surface_samples[:, self.num_near_surface_input_samples:]

		data_sample = (
			batch_uniform_input_samples.detach(),
			batch_uniform_loss_samples.detach(),
			batch_near_surface_input_samples.detach(),
			batch_near_surface_loss_samples.detach(),
			batch_surface_samples.detach()
		)

		return data_sample
