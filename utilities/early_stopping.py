import numpy as np

class EarlyStopping:
	def __init__(self, patience=10, threshold=0, improvement_callback=None):
		self.patience = patience
		self.threshold = threshold
		self.improvement_callback = improvement_callback
		self.counter = 0
		self.best_loss = None
		self.early_stop = False


	def __call__(self, val_loss, callback_args=None):
		# Initial epoch
		if self.best_loss is None:
			self.best_loss = val_loss

		# No improvement
		if val_loss > self.best_loss * (1 - self.threshold):
			self.counter += 1

		# Improvement
		else:
			self.best_loss = val_loss
			self.counter = 0

			if self.improvement_callback != None:
				if callback_args:
					self.improvement_callback(*callback_args)
				else:
					self.improvement_callback()

		# Early stop reached
		if self.counter >= self.patience:
			self.early_stop = True

