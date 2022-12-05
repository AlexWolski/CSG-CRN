import torch


# Converts a tensor of Nx4 quaternion tensors to Nx3x3 rotation matrices
# where N is the number of quaternions
def quats_to_rot_matrices(quaternions):
	# Allocate space for N rotation matrices
	N = quaternions.size(dim=0)
	matrices = quaternions.new_zeros((N, 3, 3))

	# Quaternion stored as [w, x, y, z]
	w = quaternions[:, 0]
	x = quaternions[:, 1]
	y = quaternions[:, 2]
	z = quaternions[:, 3]

	wx2 = 2*w*x
	wy2 = 2*w*y
	wz2 = 2*w*z
	xy2 = 2*x*y
	yz2 = 2*y*z
	xz2 = 2*x*z

	xx2 = 2*x*x
	yy2 = 2*y*y
	zz2 = 2*z*z

	# List of all quaternion indices
	# Technique borrowed from Superquadric Parsing Project
	# https://github.com/paschalidoud/superquadric_parsing/blob/master/learnable_primitives/primitives.py#L207-L218
	idxs = torch.arange(N).to(quaternions.device)

	# Construct rotation matrices
	matrices[idxs, 0, 0] = 1 - yy2 - zz2
	matrices[idxs, 0, 1] = xy2 - wz2
	matrices[idxs, 0, 2] = xz2 + wy2

	matrices[idxs, 1, 0] = xy2 + wz2
	matrices[idxs, 1, 1] = 1 - xx2 - zz2
	matrices[idxs, 1, 2] = yz2 - wx2

	matrices[idxs, 2, 0] = xz2 - wy2
	matrices[idxs, 2, 1] = yz2 + wx2
	matrices[idxs, 2, 2] = 1 - xx2 - yy2

	return matrices


if __name__ == "__main__":
	batch_size = 2
	num_points = 1024

	points = torch.randn([batch_size, num_points, 3])
	rotation = torch.randn([batch_size, 4])
	translation = torch.randn([batch_size, 3])
	scale = torch.randn([batch_size, 3])

	rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)

	rot_max = quats_to_rot_matrices(rotation)

	print('Quaternions:')
	print(rotation, '\n')

	print('Rotation Matrices:')
	print(rot_max)