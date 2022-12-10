import torch


# Converts a Bx4 quaternion tensor to a Bx3x3 rotation matrix tensor
# Where B = Batch size
# Where B is the number of quaternions
def quats_to_rot_matrices(quaternions):
	# Allocate space for B rotation matrices
	batch_size = quaternions.size(dim=0)
	matrices = quaternions.new_zeros((batch_size, 3, 3))

	# Quaternion stored as [w, x, y, z]
	w = quaternions[:, 0]
	x = quaternions[:, 1]
	y = quaternions[:, 2]
	z = quaternions[:, 3]

	wx2 = w*x*2
	wy2 = w*y*2
	wz2 = w*z*2
	xy2 = x*y*2
	yz2 = y*z*2
	xz2 = x*z*2

	xx2 = x*x*2
	yy2 = y*y*2
	zz2 = z*z*2

	# Construct rotation matrices
	matrices[:, 0, 0] = 1 - yy2 - zz2
	matrices[:, 0, 1] = xy2 - wz2
	matrices[:, 0, 2] = xz2 + wy2

	matrices[:, 1, 0] = xy2 + wz2
	matrices[:, 1, 1] = 1 - xx2 - zz2
	matrices[:, 1, 2] = yz2 - wx2

	matrices[:, 2, 0] = xz2 - wy2
	matrices[:, 2, 1] = yz2 + wx2
	matrices[:, 2, 2] = 1 - xx2 - yy2

	return matrices


# Transforms a BxNx3 point cloud tensor to a given space
# Where B = Batch size and N = Number of points
# Target space is defined by a Bx3 translation tensor and a Bx4 quaternion tensor
def transform_point_clouds(point_clouds, translations, rotations):
	# Translate points
	transformed_points = point_clouds - translations.unsqueeze(1)

	# Rotate points
	rot_matrices = quats_to_rot_matrices(rotations).unsqueeze(1)
	transformed_points = transformed_points.unsqueeze(-1)
	transformed_points = rot_matrices.matmul(transformed_points).squeeze(-1)

	return transformed_points


# Test transform
def test():
	batch_size = 2
	num_points = 2

	points = torch.rand([batch_size, num_points, 3])
	translations = torch.rand([batch_size, 3])
	rotations = torch.rand([batch_size, 4])

	rotations = torch.nn.functional.normalize(rotations, p=2, dim=-1)

	print('Transformed Points:')
	print(transform_point_clouds(points, translations, rotations))