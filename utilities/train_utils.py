import os
import torch
import shutil

from argparse import Namespace
from datetime import datetime as dt, timedelta
from losses.loss import Loss
from torch import autocast
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm

from networks.csg_crn import CSG_CRN
from utilities.accuracy_metrics import compute_chamfer_distance_csg_fast
from utilities.constants import SHARED_PARAMS, SEPARATE_PARAMS
from utilities.csg_model import CSGModel, add_sdf, subtract_sdf
from utilities.data_processing import get_data_files, BEST_MODEL_FILE, LATEST_MODEL_FILE
from utilities.data_augmentation import RotationAxis
from utilities.datasets import PointDataset
from utilities.early_stopping import EarlyStopping


# Prepare data files and load training dataset
def load_data_splits(args, data_split):
	# Load sample files
	file_rel_paths = get_data_files(args.data_dir, args.sub_dir)
	print(f'Found {len(file_rel_paths)} data files')

	# Split dataset
	(train_split, val_split, test_split) = torch.utils.data.random_split(file_rel_paths, data_split)

	# Ensure each dataset has enough samples
	for dataset in [('Train', train_split), ('Validation', val_split), ('Test', test_split)]:
		# Check if any dataset is empty
		if len(dataset[1].indices) == 0:
			err_msg = f'{dataset[0]} dataset is empty! Add more data samples'
			raise Exception(err_msg)

		num_samples = len(dataset[1].indices)
		num_augment_samples = num_samples * args.augment_copies

		# Check if batch size is larger than dataset size
		if not args.keep_last_batch and num_augment_samples < args.batch_size:
			err_msg = f'{dataset[0]} dataset ({num_augment_samples}) is smaller than batch size ({args.batch_size})! Add data samples or set keep_last_batch option'
			raise Exception(err_msg)

	print(f'Training set:\t{len(train_split.indices)} samples')
	print(f'Validation set:\t{len(val_split.indices)} samples')
	print(f'Testing set:\t{len(test_split.indices)} samples\n')

	return (train_split, val_split, test_split)


# Load saved settings if a model path is provided
def load_saved_settings(model_path):
	if model_path:
		torch.serialization.add_safe_globals([Namespace, Subset, RotationAxis, timedelta])
		saved_settings_dict = torch.load(model_path, weights_only=True)
		model_params = saved_settings_dict['model']
		return (saved_settings_dict, model_params)
	else:
		return (None, None)


# Load CSG-CRN network model
def load_model(num_prims, num_shapes, num_operations, device, args, model_params=None):
	predict_blending = not args.no_blending
	predict_roundness = not args.no_roundness

	# Initialize model
	model = CSG_CRN(
		num_prims,
		num_shapes,
		num_operations,
		args.num_input_points,
		args.sample_dist,
		args.surface_uniform_ratio,
		device,
		args.decoder_layers,
		not args.no_extended_input,
		predict_blending,
		predict_roundness,
		args.no_batch_norm
	)

	# Load model parameters if available
	if model_params:
		model.load_state_dict(model_params)

	return model


# Iteratively predict primitives and propagate average loss
def train_one_epoch(model, loss_func, optimizer, scaler, train_loader, num_cascades, args, device, desc='', prev_cascades_list=None):
	total_train_loss = 0

	for data_sample in tqdm(train_loader, desc=desc):
		(
			uniform_input_samples,
			uniform_loss_samples,
			near_surface_input_samples,
			near_surface_loss_samples,
			surface_samples
		) = data_sample

		input_samples = combine_and_shuffle_samples(uniform_input_samples, near_surface_input_samples)

		if args.cascade_training_mode == SEPARATE_PARAMS:
			with autocast(device_type=device.type, dtype=torch.float16, enabled=args.enable_amp):
				csg_model = model.forward_separate_cascades(input_samples.detach(), prev_cascades_list)

			cascade_loss = loss_func(near_surface_loss_samples.detach(), uniform_loss_samples.detach(), surface_samples.detach(), csg_model)
			_backpropagate(scaler, optimizer, cascade_loss)
		elif args.cascade_training_mode == SHARED_PARAMS:
			# Update model parameters after each refinement step
			if not args.backprop_all_cascades:
				csg_model = None

				for i in range(num_cascades + 1):
					if csg_model != None:
						csg_model = csg_model.detach()

					# Forward
					with autocast(device_type=device.type, dtype=torch.float16, enabled=args.enable_amp):
						csg_model = model.forward(input_samples.detach(), csg_model)

					cascade_loss = loss_func(near_surface_loss_samples.detach(), uniform_loss_samples.detach(), surface_samples.detach(), csg_model)

					# Back propagate through each cascade separately
					_backpropagate(scaler, optimizer, cascade_loss)

			# Update model parameters after all cascade iterations
			else:
				with autocast(device_type=device.type, dtype=torch.float16, enabled=args.enable_amp):
					csg_model = model.forward_cascade(input_samples.detach(), num_cascades)

				cascade_loss = loss_func(near_surface_loss_samples.detach(), uniform_loss_samples.detach(), surface_samples.detach(), csg_model)
				_backpropagate(scaler, optimizer, cascade_loss)

		# Only record the loss for the completed reconstruction
		total_train_loss += cascade_loss

	total_train_loss /= train_loader.__len__()
	return total_train_loss.cpu().item()


# Back propagate
def _backpropagate(scaler, optimizer, batch_loss):
	scaler.scale(batch_loss).backward()
	scaler.step(optimizer)
	optimizer.zero_grad(set_to_none=True)
	scaler.update()


def validate(model, loss_func, val_loader, num_cascades, args, prev_cascades_list=None):
	total_val_loss = 0
	total_chamfer_dist = 0

	# Set the model to Eval mode as to not modify the BatchNorm layer
	was_training = model.training
	model.eval()

	with torch.no_grad():
		for data_sample in val_loader:
			(
				uniform_input_samples,
				uniform_loss_samples,
				near_surface_input_samples,
				near_surface_loss_samples,
				surface_samples
			) = data_sample

			input_samples = combine_and_shuffle_samples(uniform_input_samples, near_surface_input_samples)

			if args.cascade_training_mode == SEPARATE_PARAMS:
				csg_model = model.forward_separate_cascades(input_samples, prev_cascades_list)
			else:
				csg_model = model.forward_cascade(input_samples, num_cascades)

			batch_loss = loss_func(near_surface_loss_samples, uniform_loss_samples, surface_samples, csg_model)
			total_val_loss += batch_loss.item()
			total_chamfer_dist += compute_chamfer_distance_csg_fast(surface_samples, csg_model, args.num_val_acc_points, args.val_sample_dist)

	# Return model to its previous mode
	model.train() if was_training else model.eval()

	total_val_loss /= val_loader.__len__()
	total_chamfer_dist /= val_loader.__len__()
	return (total_val_loss, total_chamfer_dist)


# Save the checkpoint of a model with shared parameters for all cascades
def save_shared_model_checkpoint(model, args, data_splits, training_logger):
	checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{training_logger.get_last_epoch()}.pt')
	save_model(model, args, data_splits, training_logger.get_results(), checkpoint_path)
	print(f'Checkpoint saved to: {checkpoint_path}\n')


# Save a model trained on a specific cascade
def save_separate_trained(model, args, data_splits, training_logger, prev_cascades_list):
	cascade_index = training_logger.get_last_cascade()
	cascade_index = cascade_index if cascade_index is not None else 0
	os.makedirs(args.cascade_models_dir, exist_ok=True)
	cascade_path = os.path.join(args.cascade_models_dir, f'cascade_{cascade_index}.pt')
	best_model_path = os.path.join(args.output_dir, BEST_MODEL_FILE)

	# Save the best model if it exists
	if os.path.isfile(best_model_path):
		shutil.copy(best_model_path, cascade_path)
	# Otherwise, save the current model
	else:
		save_model(model, args, data_splits, training_logger.get_results(), cascade_path, prev_cascades_list)

	print(f'Cascade {cascade_index} model saved to: {cascade_path}\n')

	# Return the trained model parameters for the previous cascade
	(_, prev_cascade_params) = load_saved_settings(cascade_path)
	return prev_cascade_params


# Save the model and settings to file
def save_model(model, args, data_splits, training_results, model_path, prev_cascades_list=None):
	torch.save({
		'model': model.state_dict(),
		'prev_cascades_list': prev_cascades_list,
		'args': args,
		'data_dir': args.data_dir,
		'sub_dir': args.sub_dir,
		'output_dir': args.output_dir,
		'data_splits': data_splits,
		'training_results': training_results
	}, model_path)


# Combine uniform and near-surface samples of input and loss tensors and shuffle results
def combine_and_shuffle_samples(uniform_samples, near_surface_samples):
	# Combine samples
	input_samples = torch.cat((uniform_samples, near_surface_samples), 1)
	# Shuffle samples
	input_samples = input_samples[:, torch.randperm(input_samples.size(dim=1))]
	return input_samples


def schedule_sub_weight(sub_schedule_start_epoch, sub_schedule_end_epoch, epoch):
	epoch_range = sub_schedule_end_epoch - sub_schedule_start_epoch
	sub_weight = max(0, min(1, (epoch - sub_schedule_start_epoch) / epoch_range))
	return sub_weight


# Train model for max_epochs or until stopped early
def train(model, loss_func, optimizer, scheduler, scaler, train_loader, val_loader, training_logger, data_splits, args, device):
	model.train(True)
	model.set_operation_weight(subtract_sdf, add_sdf, args.sub_weight)
	prev_cascades_list = []

	# Initalize training time ellapsed counter
	saved_time_ellapsed = training_logger.get_last_time_ellapsed()
	train_start_time = dt.now() - saved_time_ellapsed if saved_time_ellapsed else dt.now()

	# Initialize early stopper
	trained_model_path = os.path.join(args.output_dir, BEST_MODEL_FILE)
	latest_model_path = os.path.join(args.output_dir, LATEST_MODEL_FILE)
	save_best_model = lambda: save_model(model, args, data_splits, training_logger.get_results(), trained_model_path, prev_cascades_list)
	early_stopping = EarlyStopping(args.early_stop_patience, args.early_stop_threshold, save_best_model)

	# Train until model stops improving or a maximum number of epochs is reached
	init_epoch = training_logger.get_last_epoch()+1 if training_logger.get_last_epoch() else 1
	# Initialize current number of cascades from train logger
	num_cascades = training_logger.get_last_cascade() if training_logger.get_last_cascade() else 0

	for epoch in range(init_epoch, args.max_epochs+1):

		# Set operation weight
		if not args.disable_sub_operation and args.schedule_sub_weight:
			args.sub_weight = schedule_sub_weight(args.sub_schedule_start_epoch, args.sub_schedule_end_epoch, epoch)
			model.set_operation_weight(subtract_sdf, add_sdf, args.sub_weight)

		# When using the same model parameters for all cascades, use the schedule to control the number of cascades
		if args.cascade_training_mode == SHARED_PARAMS:
			# Schedule number of cascades
			if args.no_schedule_cascades:
				num_cascades = args.num_cascades
			else:
				cascade_scheduler_current = max(epoch - args.sub_schedule_start_epoch, 0) if args.schedule_sub_weight else epoch
				num_cascades = cascade_scheduler_current // args.cascade_schedule_epochs
				num_cascades = min(num_cascades, args.num_cascades)

		# Train model
		desc = f'Epoch {epoch}/{args.max_epochs}'
		train_loss = train_one_epoch(model, loss_func, optimizer, scaler, train_loader, num_cascades, args, device, desc, prev_cascades_list)
		(val_loss, chamfer_dist) = validate(model, loss_func, val_loader, num_cascades, args, prev_cascades_list)
		learning_rate = optimizer.param_groups[0]['lr']

		# Record epoch training results
		current_time = dt.now()
		time_ellapsed = current_time - train_start_time
		training_logger.add_result(epoch, num_cascades, train_loss, val_loss, chamfer_dist, learning_rate, time_ellapsed)

		weight_scheduling_in_progress = args.schedule_sub_weight and args.sub_weight < 1
		cascade_scheduling_in_progress = not args.no_schedule_cascades and num_cascades < args.num_cascades

		# Update learning rate scheduler and early stopping
		if not weight_scheduling_in_progress and not cascade_scheduling_in_progress:
			scheduler.step(chamfer_dist)
			early_stopping(chamfer_dist)

		# Print and save epoch training results
		print(f"Learning Rate:      {learning_rate}")
		print(f"Training Loss:      {train_loss}")
		print(f"Validation Loss:    {val_loss}")
		print(f"Chamfer Dist:       {chamfer_dist}")

		# Subtract weight scheduler runs first
		if weight_scheduling_in_progress:
			print(f"Subtract Weight:   {args.sub_weight}")
			print(f"Weight Scheduler:  {epoch}/{args.sub_schedule_end_epoch}")
		# Cascade scheduler runs after subtract weight scheduler completes
		elif cascade_scheduling_in_progress:
			print(f"Number of Cascades: {num_cascades}/{args.num_cascades}")
			print(f"Cascades Scheduler: {(cascade_scheduler_current) % args.cascade_schedule_epochs}/{args.cascade_schedule_epochs}")
		# Learning rate scheduling and early stopping runs after all other schedulers
		else:
			print(f"Best Chamfer Dist:  {scheduler.best}")
			print(f"LR Patience:        {scheduler.num_bad_epochs}/{scheduler.patience}")
			print(f"Early Stop:         {early_stopping.counter}/{early_stopping.patience}")

		print("Ellapsed Time:      %02d:%02d:%02d" % (time_ellapsed.seconds // 3600, time_ellapsed.seconds // 60, time_ellapsed.seconds % 60))
		print()

		# Check for early stopping
		if early_stopping.early_stop:
			separate_params_training = args.cascade_training_mode == SEPARATE_PARAMS
			last_cascade = num_cascades >= args.num_cascades

			# When training cascades separately and training completes,
			# save the model parameters and reset the training management
			if separate_params_training:
				cascade_params = save_separate_trained(model, args, data_splits, training_logger, prev_cascades_list)
				prev_cascades_list.append(cascade_params)
				early_stopping.reset()
				optimizer = init_optimizer(model, args.init_lr)
				scheduler = init_scheduler(optimizer, args)
				num_cascades += 1

			# Training is complete when the model stops improving on the last cascade
			if last_cascade:
				print(f'Stopping Training. Validation loss has not improved in {args.early_stop_patience} epochs')
				break

		# Save checkpoint parameters
		if epoch % args.checkpoint_freq == 0 and args.cascade_training_mode == SHARED_PARAMS:
				save_shared_model_checkpoint(model, args, data_splits, training_logger)

		# Save latest model parameters
		save_model(model, args, data_splits, training_logger.get_results(), latest_model_path, prev_cascades_list)

	print('\nTraining complete! Model parameters saved to:')
	print(trained_model_path)


def init_training_params(training_logger, data_splits, args, device, model_params=None):
	# Initialize model
	model = load_model(args.num_prims, CSGModel.num_shapes, CSGModel.num_operations, device, args, model_params if args.resume_training else None)
	loss_func = Loss(args.loss_metric, args.num_loss_points, args.clamp_dist, args.loss_sampling_method).to(device)
	current_lr = training_logger.get_last_lr() if training_logger.get_last_lr() else args.init_lr
	optimizer = init_optimizer(model, current_lr)
	scheduler = init_scheduler(optimizer, args)
	scaler = torch.amp.GradScaler(enabled=args.enable_amp)

	# Load training set
	(train_split, val_split, test_split) = data_splits

	if not (train_dataset := PointDataset(train_split, device, args, augment_data=args.augment_data, sampling_method=args.loss_sampling_method, dataset_name="Training Set")):
		return

	if not (val_dataset := PointDataset(val_split, device, args, augment_data=False, sampling_method=args.loss_sampling_method, dataset_name="Validation Set")):
		return

	train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=not args.keep_last_batch)
	val_sampler = BatchSampler(RandomSampler(val_dataset), batch_size=args.batch_size, drop_last=not args.keep_last_batch)

	# The PointDataset class has a custom __getitem__ function so the collate function is unneeded
	collate_fn = lambda data: data[0]
	train_loader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fn)
	val_loader = DataLoader(val_dataset, sampler=val_sampler, collate_fn=collate_fn)

	return (
		model,
		loss_func,
		optimizer,
		scheduler,
		scaler,
		train_loader,
		val_loader
	)


def init_optimizer(model, learning_rate):
	return AdamW(model.parameters(), lr=learning_rate)


def init_scheduler(optimizer, args):
	return lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience, threshold=args.lr_threshold, threshold_mode='rel')
