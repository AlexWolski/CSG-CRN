import torch
from point_transform import transform_point_clouds

import trimesh
import pyrender
import numpy as np


# Equations for all primitive SDFs are borrowed from iquilezles.org
# https://iquilezles.org/articles/distfunctions/


def sdf_sphere(query_points, translations, rotations, scales):
	radius = 1

	# Transform query points to primitive space
	transformed_query_points = transform_point_clouds(query_points, translations, rotations, scales)
	# Find distance to surface of sphere
	distances = transformed_query_points.norm(dim=-1) - radius

	return distances


def sdf_cuboid(query_points, translations, rotations, scales):
	dimensions = torch.tensor([1,1,1], dtype=float).unsqueeze(0)

	# Transform query points to primitive space
	transformed_query_points = transform_point_clouds(query_points, translations, rotations, scales)

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


def sdf_cylinder(query_points, translations, rotations, scales):
	dimensions = torch.tensor([1,1], dtype=float).unsqueeze(0)

	# Transform query points to primitive space
	transformed_query_points = transform_point_clouds(query_points, translations, rotations, scales)

	# Compute distance to uncapped cylinder
	qxy = transformed_query_points[..., :2]
	qz = transformed_query_points[..., 2]

	length = qxy.norm(dim=-1)
	d = torch.stack((length, qz), dim=-1).abs()
	d -= dimensions

	dx = d[..., 0]
	dy = d[..., 1]

	pos_distance = torch.max(d, torch.zeros_like(d)).norm(dim=-1)
	neg_distance = torch.min(torch.max(dx, dy), torch.zeros_like(dx))

	return pos_distance + neg_distance


# Test SDFs
if __name__ == "__main__":
	batch_size = 2
	num_points = 2

	points = torch.randn([batch_size, num_points, 3])
	translations = torch.tensor([0,0.2,0], dtype=float).unsqueeze(0)
	rotations = torch.tensor([0.924,0,0,0.383], dtype=float).unsqueeze(0)
	scales = torch.tensor([0.5,0.5,0.5], dtype=float).unsqueeze(0)

	distances = sdf_sphere(points, translations, rotations, scales)
	print('Sphere SDF Samples:')
	print(distances, '\n')

	distances = sdf_cuboid(points, translations, rotations, scales)
	print('Cuboid SDF Samples:')
	print(distances, '\n')

	distances = sdf_cylinder(points, translations, rotations, scales)
	print('Cylinder SDF Samples:')
	print(distances)