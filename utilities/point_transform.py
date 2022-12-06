import torch


# Converts a Bx4 quaternion tensor to a Bx3x3 rotation matrix tensor
# Where B is the number of quaternions
def quats_to_rot_matrices(quaternions):
	# Allocate space for B rotation matrices
	B = quaternions.size(dim=0)
	matrices = quaternions.new_zeros((B, 3, 3))

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
# Where B is the number of point clouds and N is the number of points in each cloud
# Target space is defined by a Bx3 translation tensor, a Bx4 quaternion tensor, and a Bx3 scale tensor
def transform_point_clouds(point_clouds, translations, rotations):
	B = point_clouds.size(dim=0)

	# Translate points
	transformed_points = point_clouds - translations.unsqueeze(1)

	# Rotate points
	rot_matrices = quats_to_rot_matrices(rotations)
	rot_matrices = rot_matrices.unsqueeze(1)
	transformed_points = transformed_points.unsqueeze(-1)
	transformed_points = rot_matrices.matmul(transformed_points).squeeze(-1)

	return transformed_points


# Test transform
if __name__ == "__main__":
	batch_size = 2
	num_points = 2

	points = torch.rand([batch_size, num_points, 3])
	translations = torch.rand([batch_size, 3])
	rotations = torch.rand([batch_size, 4])

	rotations = torch.nn.functional.normalize(rotations, p=2, dim=-1)

	print('Transformed Points:')
	print(transform_point_clouds(points, translations, rotations))