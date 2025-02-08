import os
import glob
import argparse


category_id_map = {
	'04379243': 'table',
	'03593526': 'jar',
	'04225987': 'skateboard',
	'02958343': 'car',
	'02876657': 'bottle',
	'04460130': 'tower',
	'03001627': 'chair',
	'02871439': 'bookshelf',
	'02942699': 'camera',
	'02691156': 'airplane',
	'03642806': 'laptop',
	'02801938': 'basket',
	'04256520': 'sofa',
	'03624134': 'knife',
	'02946921': 'can',
	'04090263': 'rifle',
	'04468005': 'train',
	'03938244': 'pillow',
	'03636649': 'lamp',
	'02747177': 'trash bin',
	'03710193': 'mailbox',
	'04530566': 'watercraft',
	'03790512': 'motorbike',
	'03207941': 'dishwasher',
	'02828884': 'bench',
	'03948459': 'pistol',
	'04099429': 'rocket',
	'03691459': 'loudspeaker',
	'03337140': 'file cabinet',
	'02773838': 'bag',
	'02933112': 'cabinet',
	'02818832': 'bed',
	'02843684': 'birdhouse',
	'03211117': 'display',
	'03928116': 'piano',
	'03261776': 'earphone',
	'02992529': 'telephone (redundant)',
	'04401088': 'telephone',
	'04330267': 'stove',
	'03759954': 'microphone',
	'02924116': 'bus',
	'03797390': 'mug',
	'04074963': 'remote',
	'02808440': 'bathtub',
	'02880940': 'bowl',
	'03085013': 'keyboard',
	'03467517': 'guitar',
	'04554684': 'washer',
	'02834778': 'bicycle',
	'03325088': 'faucet',
	'04004475': 'printer',
	'02954340': 'cap',
	'03046257': 'clock',
	'03513137': 'helmet',
	'03761084': 'microwave',
	'03991062': 'pot'
}


# Parse command-line arguments
def options():
	# Parsers
	help_parser = argparse.ArgumentParser(add_help=False)
	data_parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
	data_group = data_parser.add_argument_group('DATA SETTINGS')

	# Help flag
	help_parser.add_argument('-h', '--help', default=False, action='store_true', help='Print help text')

	# Data settings
	data_group.add_argument('--shapenet_dir', type=str, required=True, help='Directory of ShapeNet dataset. WARNING: Files in directory will be deleted.')


	# Parse and handle Help argument
	args, remaining_args = help_parser.parse_known_args()

	if args.help or not remaining_args:
		print()
		data_parser.print_help()
		exit()

	# Parse augment settings
	data_parser.parse_args(args=remaining_args, namespace=args)

	return args


def rename_model_files(shapenet_dir):
	"""
	Rename model files to object ID.

	Parameters
	----------
	source_dir : str
		Parent directory containing ShapeNet dataset.

	"""
	for (root, _, files) in os.walk(shapenet_dir):
		for file in files:
			# Rename model files
			if file == 'model_normalized.obj':
				file_path = os.path.join(root, file)
				model_dir = os.path.dirname(root)
				new_file_path = model_dir + '.obj'
				os.rename(file_path, new_file_path)

			# Remove unneeded files
			elif not file.endswith('.obj'):
				os.remove(os.path.join(root, file))


def remove_empty_dirs(shapenet_dir):
	"""
	Remove empty directories.

	Parameters
	----------
	source_dir : str
		Parent directory containing ShapeNet dataset.

	"""
	for (root, dirs, files) in os.walk(shapenet_dir):
		for directory in dirs:
			dir_path = os.path.join(root, directory)
			if len(os.listdir(dir_path)) == 0:
				os.rmdir(dir_path)

		if len(os.listdir(root)) == 0:
			os.rmdir(root)


def rename_category_dirs(shapenet_dir):
	"""
	Rename category folders.

	Parameters
	----------
	source_dir : str
		Parent directory containing ShapeNet dataset.

	"""
	for category_dir in os.listdir(shapenet_dir):
		if category_dir in category_id_map:
			category_name = category_id_map[category_dir]
			category_dir_path = os.path.join(shapenet_dir, category_dir)
			new_category_dir = os.path.join(shapenet_dir, category_name)
			os.rename(category_dir_path, new_category_dir)


if __name__ == '__main__':
	args = options()
	#rename_model_files(args.shapenet_dir)
	remove_empty_dirs(args.shapenet_dir)
	rename_category_dirs(args.shapenet_dir)
