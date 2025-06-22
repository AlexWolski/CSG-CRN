import os
import torch

from tqdm import tqdm
from torch import autocast
from networks.csg_crn import CSG_CRN
from utilities.csg_model import add_sdf, subtract_sdf
from utilities.data_processing import get_data_files
from utilities.early_stopping import EarlyStopping
from utilities.accuracy_metrics import compute_chamfer_distance_csg_fast


# Prepare data files and load training dataset
def load_data_splits(args, data_split, device):
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
		predict_blending,
		predict_roundness,
		args.no_batch_norm
	)

	# Load model parameters if available
	if model_params:
		model.load_state_dict(model_params)

	return model


# Iteratively predict primitives and propagate average loss
def train_one_epoch(model, loss_func, optimizer, scaler, train_loader, num_cascades, args, device, desc=''):
	total_train_loss = 0
	batch_loss = 0

	for data_sample in tqdm(train_loader, desc=desc):
		(
			target_input_samples,
			target_loss_samples,
			target_surface_samples
		) = data_sample

		csg_model = None

		# Update model parameters after each refinement step
		for i in range(num_cascades + 1):
			# Forward
			with autocast(device_type=device.type, dtype=torch.float16, enabled=not args.disable_amp):
				csg_model = csg_model.detach() if csg_model != None else csg_model
				csg_model = model.forward(target_input_samples.detach(), csg_model)

			batch_loss = loss_func(target_loss_samples.detach(), target_surface_samples.detach(), csg_model)

			# Back propagate
			scaler.scale(batch_loss).backward()
			scaler.step(optimizer)
			optimizer.zero_grad(set_to_none=True)
			scaler.update()

		# Only record the loss for the completed reconstruction
		total_train_loss += batch_loss

	total_train_loss /= train_loader.__len__()
	return total_train_loss.cpu().item()


def validate(model, loss_func, val_loader, num_cascades, args):
	total_val_loss = 0
	total_chamfer_dist = 0

	with torch.no_grad():
		for data_sample in val_loader:
			(
				target_input_samples,
				target_loss_samples,
				target_surface_samples
			) = data_sample

			csg_model = model.forward_cascade(target_input_samples, num_cascades)

			batch_loss = loss_func(target_loss_samples, target_surface_samples, csg_model)
			total_val_loss += batch_loss.item()
			total_chamfer_dist += compute_chamfer_distance_csg_fast(target_surface_samples, csg_model, args.num_val_acc_points, args.val_sample_dist)

	total_val_loss /= val_loader.__len__()
	total_chamfer_dist /= val_loader.__len__()
	return (total_val_loss, total_chamfer_dist)


# Save the model and settings to file
def save_model(model, args, data_splits, training_results, model_path):
	torch.save({
		'model': model.state_dict(),
		'args': args,
		'data_dir': args.data_dir,
		'sub_dir': args.sub_dir,
		'output_dir': args.output_dir,
		'data_splits': data_splits,
		'training_results': training_results
	}, model_path)


def schedule_sub_weight(sub_schedule_start_epoch, sub_schedule_end_epoch, epoch):
	epoch_range = sub_schedule_end_epoch - sub_schedule_start_epoch
	sub_weight = max(0, min(1, (epoch - sub_schedule_start_epoch) / epoch_range))
	return sub_weight


# Train model for max_epochs or until stopped early
def train(model, loss_func, optimizer, scheduler, scaler, train_loader, val_loader, data_splits, args, device, training_logger):
	model.train(True)
	model.set_operation_weight(subtract_sdf, add_sdf, args.sub_weight)

	# Initialize early stopper
	trained_model_path = os.path.join(args.output_dir, 'best_model.pt')
	save_best_model = lambda: save_model(model, args, data_splits, training_logger.get_results(), trained_model_path)
	early_stopping = EarlyStopping(args.early_stop_patience, args.early_stop_threshold, save_best_model)

	# Train until model stops improving or a maximum number of epochs is reached
	init_epoch = training_logger.get_last_epoch()+1 if training_logger.get_last_epoch() else 1

	for epoch in range(init_epoch, args.max_epochs+1):

		# Set operation weight
		if not args.disable_sub_operation and args.schedule_sub_weight:
			args.sub_weight = schedule_sub_weight(args.sub_schedule_start_epoch, args.sub_schedule_end_epoch, epoch)
			model.set_operation_weight(subtract_sdf, add_sdf, args.sub_weight)

		# Schedule number of cascades
		if args.no_schedule_cascades:
			num_cascades = args.num_cascades
		else:
			cascade_scheduler_current = max(epoch - args.sub_schedule_start_epoch, 0) if args.schedule_sub_weight else epoch
			num_cascades = cascade_scheduler_current // args.cascade_schedule_epochs
			num_cascades = min(num_cascades, args.num_cascades)

		# Train model
		desc = f'Epoch {epoch}/{args.max_epochs}'
		train_loss = train_one_epoch(model, loss_func, optimizer, scaler, train_loader, num_cascades, args, device, desc)
		(val_loss, chamfer_dist) = validate(model, loss_func, val_loader, num_cascades, args)
		learning_rate = optimizer.param_groups[0]['lr']

		# Record epoch training results
		training_logger.add_result(epoch, train_loss, val_loss, chamfer_dist, learning_rate)

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

		print()

		# Check for early stopping
		if early_stopping.early_stop:
			print(f'Stopping Training. Validation loss has not improved in {args.early_stop_patience} epochs')
			break

		# Save checkpoint parameters
		if epoch % args.checkpoint_freq == 0:
			checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt')
			save_model(model, args, data_splits, training_logger.get_results(), checkpoint_path)
			print(f'Checkpoint saved to: {checkpoint_path}\n')

	print('\nTraining complete! Model parameters saved to:')
	print(trained_model_path)