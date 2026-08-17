import torch
import torch.nn as nn
from torch import autocast
from utilities.csg_model import CSGModel


# Class used to split batches of training data between multiple GPUs during the training forward step.
class TrainStep(nn.Module):
	# Forward methods
	FORWARD = 'forward'
	FORWARD_CASCADE = 'forward_cascade'
	FORWARD_RESIDUAL = 'forward_residual'

	def __init__(self, model, loss_func, enable_amp):
		super(TrainStep, self).__init__()
		self.model = model
		self.loss_func = loss_func
		self.enable_amp = enable_amp

	def forward(self, near_surface_input_samples, uniform_input_samples, near_surface_loss_samples, uniform_loss_samples, surface_samples, csg_class_data=None, forward_mode=FORWARD, num_cascades=None):
		# Each tensor of training data is split by DataParallel. Reconstruct a CSGModel instance using the split tensors.
		csg_model = CSGModel.from_class_data(csg_class_data, near_surface_input_samples.size(0), near_surface_input_samples.device)

		# Run the forward pass.
		with autocast(device_type=near_surface_input_samples.device.type, dtype=torch.float16, enabled=self.enable_amp):
			if forward_mode == self.FORWARD_CASCADE:
				csg_model = self.model.forward_cascade(near_surface_input_samples, uniform_input_samples, num_cascades, csg_model)
			elif forward_mode == self.FORWARD_RESIDUAL:
				csg_model = self.model.forward_residual(near_surface_input_samples, uniform_input_samples, csg_model)
			else:
				csg_model = self.model.forward(near_surface_input_samples, uniform_input_samples, csg_model)

		return self.loss_func(near_surface_loss_samples, uniform_loss_samples, surface_samples, csg_model).unsqueeze(0)


# Helper method that calls TrainStep forward method.
def forward_train_step(train_step, near_surface_input_samples, uniform_input_samples, near_surface_loss_samples, uniform_loss_samples, surface_samples, input_csg_model=None, forward_mode=TrainStep.FORWARD, num_cascades=None):
	csg_class_data = input_csg_model.get_class_data() if input_csg_model is not None else None

	loss = train_step(
		near_surface_input_samples.detach(),
		uniform_input_samples.detach(),
		near_surface_loss_samples.detach(),
		uniform_loss_samples.detach(),
		surface_samples.detach(),
		csg_class_data,
		forward_mode,
		num_cascades
	)

	# Average the losses returned by each GPU worker.
	return loss.mean()
