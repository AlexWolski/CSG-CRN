import csv
import os
import matplotlib.pyplot as plt


HEADERS = ['Epoch', 'Training Loss', 'Validation Loss', 'Learning Rate']


class TrainingLogger():
	def __init__(self, output_folder, filename, initial_results=None):
		self.csv_output_file = os.path.join(output_folder, filename + '.csv')
		self.plot_output_file = os.path.join(output_folder, filename + '.png')
		self.training_results = initial_results if initial_results else {}

		# Initialize results structure
		if not self.training_results:
			for key in HEADERS:
				self.training_results[key] = []

		# Initialize output csv file
		self.create_csv_file()
		self.write_results(self.training_results)
		self.plot_results()


	# Create new csv file and print header
	def create_csv_file(self):
		with open(self.csv_output_file, 'w+') as fd:
			csv_writer = csv.writer(fd)
			csv_writer.writerow(HEADERS)


	# Return the list of training results
	def get_results(self):
		return self.training_results


	# Return the last recorded epoch
	def get_last_epoch(self):
		epoch_list = self.training_results['Epoch']

		if epoch_list:
			return epoch_list[-1]
		else:
			return None


	# Return the last recorded learning rate
	def get_last_lr(self):
		learning_rate = self.training_results['Learning Rate']

		if learning_rate:
			return learning_rate[-1]
		else:
			return None


	# Write all training results to csv file
	def write_results(self, initial_results):
		with open(self.csv_output_file, 'a') as fd:
			csv_writer = csv.writer(fd)

			for index in range(len(self.training_results['Epoch'])):
				epoch = self.training_results['Epoch'][index]
				train_loss = self.training_results['Training Loss'][index]
				val_loss = self.training_results['Validation Loss'][index]
				learning_rate = self.training_results['Learning Rate'][index]
				csv_writer.writerow([epoch, train_loss, val_loss, learning_rate])


	# Generate plot image of results
	def plot_results(self):
		epoch = self.training_results['Epoch']
		train_loss = self.training_results['Training Loss']
		val_loss = self.training_results['Validation Loss']
		learning_rate = self.training_results['Learning Rate']

		fig, axes1 = plt.subplots()

		axes1.plot(epoch, train_loss, color='blue', label='Training Loss')
		axes1.plot(epoch, val_loss, color='red', label='Validation Loss')
		axes1.set_xlabel('Epoch')
		axes1.set_ylabel('Loss')
		axes1.set_yscale('log')

		axes2 = axes1.twinx()
		axes2.plot(epoch, learning_rate, color='grey', label='Learning Rate', linestyle='--')
		axes1.set_ylabel('Learning Rate')

		fig.legend()
		fig.tight_layout()
		fig.savefig(self.plot_output_file, dpi=300)


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

		self.plot_results()