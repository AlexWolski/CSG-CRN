import torch
from utilities.point_transform import transform_point_cloud_batch, invert_translation, invert_quaternion


# Transform batch of query points in world space to the local space of a primitive
def world_to_local_points(query_points, translations, rotations):
	inverse_translations = invert_translation(translations)
	inverse_rotations = invert_quaternion(rotations)
	return transform_point_cloud_batch(query_points, inverse_translations, inverse_rotations)


# Equations for all primitive SDFs are borrowed from iquilezles.org
# https://iquilezles.org/articles/distfunctions/


def sdf_ellipsoid_transformed(query_points, translations, rotations, dimensions, *_):
	transformed_query_points = world_to_local_points(query_points, translations, rotations)
	return sdf_ellipsoid(query_points, dimensions)


def sdf_ellipsoid(query_points, dimensions, *_):
	dimensions = dimensions.unsqueeze(1)

	# Scale sphere to approximate ellipsoid
	k0 = (query_points / dimensions).norm(dim=-1)
	# Divide by gradient to minimize distortion
	k1 = (query_points / (dimensions*dimensions)).norm(dim=-1)
	distances = k0 * (k0 - 1.0) / k1

	return distances


def sdf_cuboid_transformed(query_points, dimensions, roundness):
	transformed_query_points = world_to_local_points(query_points, translations, rotations)
	return sdf_cuboid(transformed_query_points, dimensions, roundness)


def sdf_cuboid(query_points, dimensions, roundness):
	# Adjust roundness value per dimension to keep the rounding effect uniform
	(min_dims, _) = torch.min(dimensions, dim=-1, keepdim=True)
	adjusted_roundness = roundness * min_dims
	# Adjust the dimensions to compensate for the shrinking caused by rounding
	adjusted_dimensions = dimensions - adjusted_roundness
	adjusted_dimensions = adjusted_dimensions.unsqueeze(1)

	# Reflect query point in all quadrants and translate relative to the box surface
	query_points = query_points.abs() - adjusted_dimensions

	# Compute positive distances from outside the box
	pos_distance = query_points.clamp(min=0.0).norm(dim=-1)

	# Compute negative distances from inside the box
	qx = query_points[..., 0]
	qy = query_points[..., 1]
	qz = query_points[..., 2]
	neg_distance = qy.max(qz).max(qx).clamp(max=0.0)

	return pos_distance + neg_distance - adjusted_roundness


def sdf_cylinder_transformed(query_points, translations, rotations, dimensions, roundness):
	transformed_query_points = world_to_local_points(query_points, translations, rotations)
	return sdf_cylinder(transformed_query_points, dimensions, roundness)


def sdf_cylinder(query_points, dimensions, roundness):
	dimensions = dimensions.unsqueeze(1)

	qxy = query_points[..., :2]
	qz = query_points[..., 2]
	sxy = dimensions[..., :2]
	sz = dimensions[..., 2]

	# Compute roundness value
	adjusted_roundness = roundness * sz
	adjusted_height = sz - adjusted_roundness

	# Scale circle to approximate distance to ellipse
	k0 = (qxy / sxy).norm(dim=-1)
	# Divide by gradient to minimize distortion
	k1 = (qxy / (sxy*sxy)).norm(dim=-1)
	ellipse_distance = k0 * (k0 - 1.0) / k1 + adjusted_roundness

	# Compute distance to ends of cylinder
	cap_distance = qz.abs() - adjusted_height

	distance = torch.stack((ellipse_distance, cap_distance), dim=-1)
	dx = distance[..., 0]
	dy = distance[..., 1]

	pos_distance = torch.clamp(distance, min=0.0).norm(dim=-1)
	neg_distance = torch.clamp(torch.max(dx, dy), max=0.0)

	return pos_distance + neg_distance - adjusted_roundness


# Test SDFs
def test():
	from torch.distributions.uniform import Uniform
	
	batch_size = 2
	num_points = 2
	
	points = Uniform(-0.5, 0.5).sample((batch_size, num_points, 3))
	translations = torch.tensor([0,0.2,0], dtype=float).repeat(batch_size,1)
	rotations = torch.tensor([0.924,0,0,0.383], dtype=float).repeat(batch_size,1)
	scales = torch.tensor([0.2,0.5,0.7], dtype=float).repeat(batch_size,1)
	roundness = torch.tensor([1], dtype=float).repeat(batch_size,1)

	distances = sdf_ellipsoid(points, translations, rotations, scales)
	print('Sphere SDF Samples:')
	print(distances, '\n')

	distances = sdf_cuboid(points, translations, rotations, scales, roundness)
	print('Cuboid SDF Samples:')
	print(distances, '\n')

	distances = sdf_cylinder(points, translations, rotations, scales, roundness)
	print('Cylinder SDF Samples:')
	print(distances)