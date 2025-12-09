import csv
import os
import matplotlib.pyplot as plt


HEADERS = ['Epoch', 'Cascades', 'Training Loss', 'Validation Loss', 'Chamfer Distance', 'Learning Rate']


class TrainingLogger():
	def __init__(self, output_folder, filename, loss_metric='', initial_results=None):
		self.csv_output_file = os.path.join(output_folder, filename + '.csv')
		self.plot_output_file = os.path.join(output_folder, filename + '.png')
		self.training_results = initial_results if initial_results else {}
		self.loss_metric = loss_metric

		# Initialize results structure
		if not self.training_results:
			for key in HEADERS:
				self.training_results[key] = []

		# Initialize output csv file
		self.create_csv_file()
		self.write_results()
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


	# Return the last recorded number of cascades
	def get_last_cascade(self):
		cascade_list = self.training_results['Cascades']

		if cascade_list:
			return cascade_list[-1]
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
	def write_results(self):
		with open(self.csv_output_file, 'a') as fd:
			csv_writer = csv.writer(fd)

			for index in range(len(self.training_results['Epoch'])):
				epoch = self.training_results['Epoch'][index]
				train_loss = self.training_results['Training Loss'][index]
				val_loss = self.training_results['Validation Loss'][index]
				val_acc = self.training_results['Chamfer Distance'][index]
				learning_rate = self.training_results['Learning Rate'][index]
				csv_writer.writerow([epoch, train_loss, val_loss, val_acc, learning_rate])


	# Generate plot image of results
	def plot_results(self):
		epoch = self.training_results['Epoch']
		num_cascades = self.training_results['Cascades']
		train_loss = self.training_results['Training Loss']
		val_loss = self.training_results['Validation Loss']
		val_acc = self.training_results['Chamfer Distance']
		learning_rate = self.training_results['Learning Rate']

		fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(5,10))

		ax1.plot(epoch, train_loss, color='blue', label='Training Loss')
		ax1.plot(epoch, val_loss, color='red', label='Validation Loss')
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel(f'{self.loss_metric} Loss')
		ax1.set_yscale('log')
		ax1.legend()

		ax2.plot(epoch, val_acc, color='red', label='Validation Chamfer Distance')
		ax2.set_xlabel('Epoch')
		ax2.set_ylabel('Chamfer Distance')
		ax2.set_yscale('log')
		ax2.legend()

		ax3.plot(epoch, learning_rate, color='black', label='Learning Rate')
		ax3.set_xlabel('Epoch')
		ax3.set_ylabel('Learning Rate')
		ax3.set_yscale('log')

		ax4.plot(epoch, num_cascades, color='black', label='Cascades')
		ax4.set_xlabel('Epoch')
		ax4.set_ylabel('Cascades')
		ax4.set_yticks(range(0, 1 if not num_cascades else max(num_cascades) + 1))
		ax4.set_ylim(bottom=0)

		fig.savefig(self.plot_output_file, bbox_inches="tight", dpi=300)
		plt.close()


	# Append epoch result data to training_results dictionary
	def _append_training_results(self, epoch, num_cascades, train_loss, val_loss, val_acc, learning_rate):
		self.training_results['Epoch'].append(epoch)
		self.training_results['Cascades'].append(num_cascades)
		self.training_results['Training Loss'].append(train_loss)
		self.training_results['Validation Loss'].append(val_loss)
		self.training_results['Chamfer Distance'].append(val_acc)
		self.training_results['Learning Rate'].append(learning_rate)


	# Log epoch training results
	def add_result(self, epoch, num_cascades, train_loss, val_loss, val_acc, learning_rate):
		self._append_training_results(epoch, num_cascades, train_loss, val_loss, val_acc, learning_rate)

		with open(self.csv_output_file, 'a') as fd:
			csv_writer = csv.writer(fd)
			csv_writer.writerow([epoch, num_cascades, train_loss, val_loss, val_acc, learning_rate])

		self.plot_results()
