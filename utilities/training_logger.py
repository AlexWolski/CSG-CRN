import csv
import os


class TrainingLogger():
	def __init__(self, output_folder, file_name, overwrite=False):
		row_headers = ['Epoch', 'Training Loss', 'Validation Loss', 'Learning Rate']
		self.csv_output_file = os.path.join(output_folder, file_name + '.csv')
		self.plot_output_file = os.path.join(output_folder, file_name + '.png')
		self.training_telemetry = {}

		# Initialize output csv file
		if not os.path.isfile(self.csv_output_file) or overwrite:
			with open(self.csv_output_file, 'w+') as fd:
				csv_writer = csv.writer(fd)
				csv_writer.writerow(row_headers)

		# Initialize results dictionary
		for key in row_headers:
			self.training_telemetry[key] = []


	# Add epoch training results to file
	def append_training_result(self, epoch, train_loss, val_loss, learning_rate):
		self.training_telemetry['Epoch'].append(epoch)
		self.training_telemetry['Training Loss'].append(train_loss)
		self.training_telemetry['Validation Loss'].append(val_loss)
		self.training_telemetry['Learning Rate'].append(learning_rate)

		with open(self.csv_output_file, 'a') as fd:
			csv_writer = csv.writer(fd)
			csv_writer.writerow([epoch, train_loss, val_loss, learning_rate])