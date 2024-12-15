import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utilities.data_processing import pre_process_sample
from utilities.data_augmentation import augment_sample_batch


class PointDataset(Dataset):
	def __init__(self, file_rel_paths, device, args, dataset_name="Dataset"):
		self.raw_copies = len(file_rel_paths)
		self.device = device
		self.augmented_copies = len(file_rel_paths) * args.augment_copies
		self.args = args

		# Load all data samples into memory
		sdf_sample_list = []
		skipped_samples = 0
		total_points = args.num_input_points + args.num_loss_points

		for file_rel_path in tqdm(file_rel_paths, desc=f'Loading {dataset_name}'):
			sample_path = os.path.join(self.args.data_dir, file_rel_path)
			sdf_sample = np.load(sample_path).astype(np.float32)

			# Preprocess sample if needed
			if not args.skip_preprocess:
				sdf_sample = pre_process_sample(args, sdf_sample)

				if sdf_sample is None:
					skipped_samples += 1
					continue

			# Ensure that the sample has the correct number of points
			if sdf_sample.shape[0] != total_points:
					skipped_samples += 1
					continue

			sdf_sample = torch.from_numpy(sdf_sample)
			sdf_sample_list.append(sdf_sample)


		if not sdf_sample_list:
			self.sdf_samples = None
			print(f'{dataset_name} contains no samples with a sufficient number of points.')
			print('Lower the number of input and loss points or use a dataset with more samples points.')
			return

		# Remove skipped copies
		self.raw_copies -= skipped_samples

		# Save samples in system memory
		self.sdf_samples = torch.stack(sdf_sample_list, dim=0)


		if skipped_samples > 0:
			print(f'Skipped {skipped_samples} samples that had too few points\n')

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

		# TODO: Implement initial recon loading and augmentation (noise injection)
		batch_recon_input_samples = None
		batch_recon_loss_samples = None

		return (batch_target_input_samples, batch_target_loss_samples, batch_recon_input_samples, batch_recon_loss_samples)
