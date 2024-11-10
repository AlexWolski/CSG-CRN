import csv
import os

HEADERS = ['Epoch', 'Training Loss', 'Validation Loss', 'Learning Rate']


class TrainingLogger():
	def __init__(self, output_folder, filename, overwrite=False):
		self.csv_output_file = os.path.join(output_folder, filename + '.csv')
		self.plot_output_file = os.path.join(output_folder, filename + '.png')
		self.training_results = {}

		# Initialize results dictionary
		for key in HEADERS:
			self.training_results[key] = []

		# Create output csv file
		if not os.path.isfile(self.csv_output_file) or overwrite:
			self._create_csv_file()
		# Load past results from csv file
		else:
			self._read_csv_file(self.csv_output_file)


	# Create new csv file and print header
	def _create_csv_file(self):
		with open(self.csv_output_file, 'w+') as fd:
			csv_writer = csv.writer(fd)
			csv_writer.writerow(HEADERS)


	# Read existing results from file
	def _read_csv_file(self, file_path):
		with open(file_path, 'r') as fd:
			for i, line in enumerate(csv.reader(fd)):
				if i > 0:
					self._append_training_results(*line)


	# Append epoch result data to training_results dictionary
	def _append_training_results(self, epoch, train_loss, val_loss, learning_rate):
		self.training_results['Epoch'].append(epoch)
		self.training_results['Training Loss'].append(train_loss)
		self.training_results['Validation Loss'].append(val_loss)
		self.training_results['Learning Rate'].append(learning_rate)


	# Log epoch training results
	def add_result(self, epoch, train_loss, val_loss, learning_rate):
		self._append_training_results(epoch, train_loss, val_loss, learning_rate)

		with open(self.csv_output_file, 'a') as fd:
			csv_writer = csv.writer(fd)
			csv_writer.writerow([epoch, train_loss, val_loss, learning_rate])
