import torch
from point_transform import transform_point_clouds


# Equations for all primitive SDFs are borrowed from iquilezles.org
# https://iquilezles.org/articles/distfunctions/


def sdf_ellipsoid(query_points, translations, rotations, dimensions):
	# Transform query points to primitive space
	transformed_query_points = transform_point_clouds(query_points, translations, rotations)

	# Scale sphere to approximate ellipsoid
	k0 = (transformed_query_points / dimensions).norm(dim=-1)
	# Divide by gradient to minimize distortion
	k1 = (transformed_query_points / (dimensions*dimensions)).norm(dim=-1)
	distances = k0 * (k0 - torch.ones_like(k0)) / k1

	return distances


def sdf_cuboid(query_points, translations, rotations, dimensions):
	# Transform query points to primitive space
	transformed_query_points = transform_point_clouds(query_points, translations, rotations)

	# Reflect query point in all quadrants and translate relative to the box surface
	transformed_query_points = transformed_query_points.abs() - dimensions

	# Compute positive distances from outside the box
	pos_distance = transformed_query_points.max(torch.zeros_like(transformed_query_points)).norm(dim=-1)

	# Compute negative distances from inside the box
	qx = transformed_query_points[..., 0]
	qy = transformed_query_points[..., 1]
	qz = transformed_query_points[..., 2]
	neg_distance = qy.max(qz).max(qx).min(torch.zeros_like(qx))

	return pos_distance + neg_distance


def sdf_cylinder(query_points, translations, rotations, dimensions):
	# Transform query points to primitive space
	transformed_query_points = transform_point_clouds(query_points, translations, rotations)

	qxy = transformed_query_points[..., :2]
	qz = transformed_query_points[..., 2]
	sxy = dimensions[..., :2]
	sz = dimensions[..., 2]

	# Scale circle to approximate distance to ellipse
	k0 = (qxy / sxy).norm(dim=-1)
	# Divide by gradient to minimize distortion
	k1 = (qxy / (sxy*sxy)).norm(dim=-1)
	ellipse_distance = k0 * (k0 - torch.ones_like(k0)) / k1

	# Compute distance to ends of cylinder
	cap_distance = qz.abs() - sz

	distance = torch.stack((ellipse_distance, cap_distance), dim=-1)
	dx = distance[..., 0]
	dy = distance[..., 1]

	pos_distance = torch.max(distance, torch.zeros_like(distance)).norm(dim=-1)
	neg_distance = torch.min(torch.max(dx, dy), torch.zeros_like(dx))

	return pos_distance + neg_distance


# Test SDFs
if __name__ == "__main__":
	batch_size = 2
	num_points = 2

	points = torch.rand([batch_size, num_points, 3])
	translations = torch.tensor([0,0.2,0], dtype=float).unsqueeze(0)
	rotations = torch.tensor([0.924,0,0,0.383], dtype=float).unsqueeze(0)
	scales = torch.tensor([0.2,0.5,0.7], dtype=float).unsqueeze(0)

	distances = sdf_ellipsoid(points, translations, rotations, scales)
	print('Sphere SDF Samples:')
	print(distances, '\n')

	distances = sdf_cuboid(points, translations, rotations, scales)
	print('Cuboid SDF Samples:')
	print(distances, '\n')

	distances = sdf_cylinder(points, translations, rotations, scales)
	print('Cylinder SDF Samples:')
	print(distances)