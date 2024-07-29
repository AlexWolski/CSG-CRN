import torch
import math


# Converts a Bx4 quaternion tensor to a Bx4x4 rotation matrix tensor
# Where B = Batch size
# https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
def quat_to_mat4_batch(quaternions):
	# Allocate space for B rotation matrices
	batch_size = quaternions.size(dim=0)
	matrices = quaternions.new_zeros((batch_size, 4, 4)).to(quaternions.device)

	# Quaternion stored as [w, x, y, z]
	w = quaternions[:, 0]
	x = quaternions[:, 1]
	y = quaternions[:, 2]
	z = quaternions[:, 3]

	# Precompute repeated terms
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

	matrices[:, 3, 3] = 1

	return matrices


# Converts a Bx3 translation tensor to a Bx4x4 translation matrix tensor
# Where B = Batch size
def translation_to_mat4_batch(translations):
	(batch_size, _) = translations.size()
	translation_transpose = translations.unsqueeze(-1)

	# Bx4x4 identity matrix
	translation_matrix = torch.eye(4).repeat(batch_size, 1, 1).to(translations.device)
	# Copy the translation to identity matrix
	translation_matrix[:,:3,3:4] = translation_transpose

	return translation_matrix


# Converts a Bx3 scale tensor to a Bx4x4 scale matrix tensor
# Where B = Batch size
def scale_to_mat4_batch(scales):
	(batch_size, _) = scales.size()

	# Add a new column of ones to form a Bx4 tensor
	new_col = torch.ones((batch_size, 1), dtype=torch.float).to(scales.device)
	scales = torch.cat((scales, new_col), dim=1)

	# Copy the scale tensor to the diagonal of a Bx4x4 tensor 
	scale_matrix = torch.diag_embed(scales)

	return scale_matrix


# Convert a point cloud from a BxNx3 cartesian coordinate system to a BxNx4 homogeneous coordinate system
def to_homogeneous_batch(point_clouds):
	(batch_size, rows, cols) = point_clouds.size()

	# Add new column of ones
	new_col = torch.ones((batch_size, rows, 1), dtype=torch.float).to(point_clouds.device)
	point_clouds_homo = torch.cat((point_clouds, new_col), dim=2)

	# Transpose so that each column represents a point
	point_clouds_homo = point_clouds_homo.transpose(1,2)

	return point_clouds_homo


# Convert a point cloud from a BxNx4 homogeneous coordinate system to a BxNx3 cartesian coordinate system
def to_cartesian_batch(point_clouds):
	point_clouds = point_clouds[:,:-1,:]
	point_clouds = point_clouds.transpose(1,2)
	return point_clouds


# Translates a BxNx3 point cloud tensor by a Bx3 translation tensor
# Where B = Batch size and N = Number of points
def translate_point_cloud_batch(point_clouds, translations):
	point_clouds_homo = to_homogeneous_batch(point_clouds)
	translation_matrices = translation_to_mat4_batch(translations)
	translated_points = translation_matrices.matmul(point_clouds_homo)
	translated_points = to_cartesian_batch(translated_points)
	return translated_points


# Translates an Nx3 point cloud matrix by a translation vector
def translate_point_cloud(point_cloud, translation):
	transformed_points = translate_point_cloud_batch(point_cloud.unsqueeze(0), translation.unsqueeze(0))
	return transformed_points.squeeze(0)


# Rotates a BxNx3 point cloud tensor by a Bx4 rotation tensor
# Where B = Batch size and N = Number of points
def rotate_point_cloud_batch(point_clouds, rotations):
	point_clouds_homo = to_homogeneous_batch(point_clouds)
	rot_matrices = quat_to_mat4_batch(rotations)
	rotated_points = rot_matrices.matmul(point_clouds_homo)
	rotated_points = to_cartesian_batch(rotated_points)
	return rotated_points


# Rotates an Nx3 point cloud matrix by a rotation vector
def rotate_point_cloud(point_cloud, rotation):
	rotated_points = rotate_point_cloud_batch(point_cloud.unsqueeze(0), rotation.unsqueeze(0))
	return rotated_points.squeeze(0)


# Scales a BxNx3 point cloud tensor by a Bx3 rotation tensor
# Where B = Batch size and N = Number of points
def scale_point_cloud_batch(point_clouds, scales):
	point_clouds_homo = to_homogeneous_batch(point_clouds)
	scale_matrices = scale_to_mat4_batch(scales)
	scaled_points = scale_matrices.matmul(point_clouds_homo)
	scaled_points = to_cartesian_batch(scaled_points)
	return scaled_points


# Scales an Nx3 point cloud matrix by a scale vector
def scale_point_cloud(point_cloud, scale):
	scaled_points = scale_point_cloud_batch(point_cloud.unsqueeze(0), scale.unsqueeze(0))
	return scaled_points.squeeze(0)


# Transforms a BxNx3 point cloud tensor to a given space
# Where B = Batch size and N = Number of points
# Target space is defined by a Bx3 translation tensor and a Bx4 quaternion tensor
# Rotations are quaternions of form [w, x, y, z]
def transform_point_cloud_batch(point_clouds, translations, rotations, scales):
	transformed_points = rotate_point_cloud_batch(point_clouds, rotations)
	transformed_points = translate_point_cloud_batch(transformed_points, translations)
	return transformed_points


# Transforms a Nx3 point cloud tensor to a given space
# Rotations are quaternions of form [w, x, y, z]
def transform_point_cloud(point_cloud, translation, rotation, scales):
	transformed_points = rotate_point_cloud(point_clouds, rotation)
	transformed_points = translate_point_cloud(transformed_points, translation)
	return transformed_points


# Test transform
def test():
	batch_size = 2
	num_points = 3

	batch_points = torch.FloatTensor([
		[[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],

        [[1.0, 0.0, 0.0],
         [1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0]]])

	batch_translations = torch.FloatTensor([
		[0, 0, 2],
        [2, 2, 2]])

	batch_rotations = torch.FloatTensor([
		[0.7071068, 0.7071068, 0, 0],
        [0.7325378, 0.4619398, 0.1913417, 0.4619398]])

	batch_scales = torch.FloatTensor([
		[0.5, 0.5, 0.5],
		[2, 2, 2]])

	expected_result = torch.FloatTensor([
		[[1.0, 0.0, 2.0],
         [0.0, 0.0, 3.0],
         [0.0, -1.0, 2.0]],

        [[2.5, 2.853553, 2.146447],
         [2, 3, 3],
         [2.707107, 2.5, 3.5]]])

	actual_result = transform_point_cloud_batch(batch_points, batch_translations, batch_rotations, batch_scales)

	print('Points:')
	print(batch_points)

	print('Translations:')
	print(batch_translations)

	print('Rotations:')
	print(batch_rotations)

	print('Transformed Batch Points:')
	print(actual_result)

	print('Expected Result:')
	print(expected_result)

	if torch.allclose(expected_result, actual_result, rtol=1e-03, atol=1e-03):
		print('Matrices match')
	else:
		print('Matrices do not match')

if __name__ == "__main__":
    test()