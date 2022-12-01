import numpy as np
from torch.utils.data import Dataset


class PointDataset(Dataset):
	def __init__(self, data_dir, filenames):
		self.data_dir = data_dir
		self.filenames = filenames

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		filename = self.filenames[idx]
		points_path = os.path.join(self.data_dir, filename)
		points = np.load(points_path)
		return points