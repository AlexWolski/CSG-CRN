import torch


# Converts a Bx4 quaternion tensor to a Bx3x3 rotation matrix tensor
# Where B = Batch size
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


# Translates a BxNx3 point cloud tensor by a Bx3 translation tensor
# Where B = Batch size and N = Number of points
def translate_point_clouds(point_clouds, translations):
	return point_clouds - translations.unsqueeze(1)


# Translates an Nx3 point cloud matrix by a translation vector
def translate_point_cloud(point_cloud, translation):
	point_cloud = point_cloud.unsqueeze(-1)
	translation = translation.unsqueeze(-1)
	transformed_points = translate_point_clouds(point_cloud, translation)
	return translate_point_clouds.squeeze(-1)


# Rotates a BxNx3 point cloud tensor by a Bx4 rotation tensor
# Where B = Batch size and N = Number of points
def rotate_point_clouds(point_clouds, rotations):
	rot_matrices = quats_to_rot_matrices(rotations).unsqueeze(1)
	rotated_points = point_clouds.unsqueeze(-1)
	rotated_points = rot_matrices.matmul(rotated_points).squeeze(-1)
	return rotated_points


# Rotates an Nx3 point cloud matrix by a rotation vector
def rotate_point_cloud(point_cloud, rotation):
	point_cloud = point_cloud
	rotation = rotation.unsqueeze(-1).transpose(0, 1)
	rotated_points = rotate_point_clouds(point_cloud, rotation).squeeze(0)
	return rotated_points


# Transforms a BxNx3 point cloud tensor to a given space
# Where B = Batch size and N = Number of points
# Target space is defined by a Bx3 translation tensor and a Bx4 quaternion tensor
def transform_point_clouds(point_clouds, translations, rotations):
	transformed_points = translate_point_clouds(point_clouds, translations)
	transformed_points = rotate_point_clouds(point_clouds, rotations)
	return transformed_points


# Transforms a Nx3 point cloud tensor to a given space
def transform_point_cloud(point_cloud, translation, rotation):
	transformed_points = translate_point_cloud(point_cloud, translation)
	transformed_points = rotate_point_cloud(point_cloud, rotation)
	return transformed_points


# Test transform
def test():
	batch_size = 2
	num_points = 2

	batch_points = torch.rand([batch_size, num_points, 3])
	batch_translations = torch.rand([batch_size, 3])
	batch_rotations = torch.rand([batch_size, 4])
	batch_rotations = torch.nn.functional.normalize(batch_rotations, p=2, dim=-1)

	print(batch_translations.shape())

	print('Transformed Batch Points:')
	print(transform_point_clouds(batch_points, batch_translations, batch_rotations))

	points = torch.rand([num_points, 3])
	translations = torch.rand([3])
	rotations = torch.rand([4])
	rotations = torch.nn.functional.normalize(rotations, p=2, dim=-1)

	print('Transformed Points:')
	print(transform_point_cloud(points, translations, rotations))

if __name__ == "__main__":
    test()