import torch

# Determine device to train on
def get_device(device=None, cpu_allowed=False):
	# Only allow CPU device when cpu_allowed is False
	if not cpu_allowed:
		if not torch.cuda.is_available():
			raise Exception('Only CUDA devices are supported. A CUDA device is required.')
		elif device == 'cpu':
			raise Exception('Only CUDA devices are supported. Select a CUDA device:' + get_available_devices(cpu_allowed))

	# When a device is not specified, automatically select one
	if not device:
		if torch.cuda.is_available():
			return torch.device('cuda')
		elif cpu_allowed:
			return torch.device('cpu')

	# Attempt to load the specified device
	try:
		return torch.device(device)
	# Display available devices
	except:
		raise Exception(f'Device {device} does not exist. Select a device: ' + get_available_devices(cpu_allowed))


# Determine list of devices to train on
def get_devices(devices=None, cpu_allowed=False):
	# When no devices are not specified, automatically select one
	if not devices:
		return [get_device(None, cpu_allowed)]

	# Convert string parameter to list
	if isinstance(devices, str):
		devices = [devices]

	resolved_devices = []
	has_cpu_device = False
	has_cuda_device = False

	# Resolve all devices
	for device in devices:
		resolved_devices.append(get_device(device, cpu_allowed))
		has_cpu_device = True if device == 'cpu' else has_cpu_device
		has_cuda_device = True if device.startswith('cuda') else has_cuda_device

	# Check for invalid comnbinations
	if has_cpu_device and has_cuda_device:
		raise Exception('Cannot use both CPU and CUDA devices together')

	return resolved_devices


# Return a list of available training devices
def get_available_devices(cpu_allowed):
	if not torch.cuda.is_available() and not cpu_allowed:
		return 'No CUDA devices are available. A CUDA device is required.'

	num_cuda = torch.cuda.device_count()
	cpu_device = '"CPU"' if cpu_allowed else ''

	if num_cuda == 0:
		cuda_deivce = ''
	elif num_cuda == 1:
		cuda_deivce = '"cuda:0"'
	else:
		cuda_deivce = f'"cuda:0"-"cuda:{torch.cuda.device_count()-1}"'

	connector = ' or ' if num_cuda and cpu_device else ''

	return cpu_device + connector + cuda_deivce
