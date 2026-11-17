import argparse
import glob
import math
import os
import shutil
import subprocess
import cv2
import numpy as np
import point_cloud_utils as pcu
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from utilities.data_processing import NEAR_SURFACE_FOLDER, SAMPLE_LIST_FILE, SURFACE_FOLDER, UNIFORM_FOLDER, save_list
from utilities.file_utils import init_dataset
from utilities.sampler_utils import sample_uniform_points_sphere

PROGRAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Programs', 'colmap', 'bin')
COLMAP_FOLDER_SUFFIX = ' COLMAP'
UP_AXIS = np.array([0.0, 1.0, 0.0])


def options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', type=str, required=True, help='Video file of an object standing on a flat surface')
	parser.add_argument('--output_dir', type=str, required=True, help='Output directory in which a dataset directory named after the video is created to store the SDF samples of the scan')
	parser.add_argument('--num_sdf_samples', type=int, default=16384, help='Number of uniform and near-surface SDF samples to generate')
	parser.add_argument('--num_surface_samples', type=int, default=30000, help='Number of surface samples to generate for accuracy computation')
	parser.add_argument('--sample_dist', type=float, default=0.1, help='Maximum distance to object surface for near-surface sampling (must be > 0)')
	parser.add_argument('--yaw', type=float, default=0.0, help='Rotation in degrees around the up axis. At 0 the side of the object facing the camera in the first frame becomes the front. The object is then turned by up to 45 degrees to align its footprint with the X and Z axes. ShapeNet objects face -Z.')
	parser.add_argument('--num_frames', type=int, default=180, help='Number of frames to reconstruct the point cloud from')
	parser.add_argument('--max_image_size', type=int, default=1600, help='Maximum width or height of the images used for dense reconstruction')
	parser.add_argument('--crop_radius', type=float, default=0.7, help='Radius of the sphere around the point the cameras look at that is kept, as a fraction of the distance to the cameras')
	parser.add_argument('--floor_tolerance', type=float, default=0.01, help='Maximum distance between the floor plane and the points used to fit it, as a fraction of the point cloud size')
	parser.add_argument('--floor_max_angle', type=float, default=30.0, help='Maximum angle in degrees between the floor normal and the up direction of the cameras')
	parser.add_argument('--outlier_neighbors', type=int, default=16, help='Number of neighbors used to measure the local point spacing when removing outliers')
	parser.add_argument('--outlier_std', type=float, default=2.0, help='Points whose spacing is this many standard deviations above the mean are removed')
	parser.add_argument('--cluster_cell_size', type=float, default=0.02, help='Points closer than this fraction of the point cloud size belong to the same cluster')
	parser.add_argument('--min_cluster_ratio', type=float, default=0.05, help='Only the cluster closest to the point the cameras look at is kept. Clusters smaller than this fraction of the largest cluster are ignored.')
	parser.add_argument('--voxel_size', type=float, default=0.006, help='Spacing of the cleaned point cloud after the object is scaled to a unit sphere')
	parser.add_argument('--device', type=int, default=0, help='Index of the CUDA device used by COLMAP for dense reconstruction')
	parser.add_argument('--colmap_path', type=str, default=os.path.join(PROGRAMS_DIR, 'colmap'), help='Path to the COLMAP executable')
	parser.add_argument('--ffmpeg_path', type=str, default=os.path.join(PROGRAMS_DIR, 'ffmpeg'), help='Path to the FFmpeg executable')

	args = parser.parse_args()
	return args


def image_sharpness(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	scale = 640 / max(image.shape)
	image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
	return cv2.Laplacian(image, cv2.CV_64F).var()


def extract_frames(video_path, frames_dir, num_frames, ffmpeg_path):
	os.makedirs(frames_dir)
	command = [ffmpeg_path, '-loglevel', 'error', '-i', video_path, '-fps_mode', 'passthrough', '-qscale:v', '2', os.path.join(frames_dir, '%05d.jpg')]
	subprocess.run(command, check=True)

	frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
	group_size = math.ceil(len(frame_paths) / num_frames)

	for start in range(0, len(frame_paths), group_size):
		candidates = frame_paths[start:start + group_size]
		sharpest = max(candidates, key=image_sharpness)

		for candidate in candidates:
			if candidate != sharpest:
				os.remove(candidate)


def run_colmap(colmap_path, log_path, command, **command_args):
	arguments = [colmap_path, command]

	for (name, value) in command_args.items():
		arguments.extend(['--' + name.replace('__', '.'), str(value)])

	print(f'COLMAP {command}')

	with open(log_path, 'a') as log_file:
		subprocess.run(arguments, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def largest_sparse_model(sparse_dir):
	model_dirs = glob.glob(os.path.join(sparse_dir, '*'))
	return max(model_dirs, key=lambda model_dir: os.path.getsize(os.path.join(model_dir, 'images.bin')))


def reconstruct_point_cloud(args, colmap_dir, cameras_dir, fused_path):
	frames_dir = os.path.join(colmap_dir, 'frames')
	sparse_dir = os.path.join(colmap_dir, 'sparse')
	dense_dir = os.path.join(colmap_dir, 'dense')
	database_path = os.path.join(colmap_dir, 'database.db')
	log_path = os.path.join(colmap_dir, 'colmap.log')

	shutil.rmtree(colmap_dir, ignore_errors=True)
	os.makedirs(sparse_dir)
	os.makedirs(cameras_dir)
	extract_frames(args.input_file, frames_dir, args.num_frames, args.ffmpeg_path)
	print(f'Reconstructing from {len(os.listdir(frames_dir))} images')

	run_colmap(
		args.colmap_path, log_path, 'feature_extractor', database_path=database_path, image_path=frames_dir, ImageReader__single_camera=1,
		FeatureExtraction__use_gpu=0, SiftExtraction__peak_threshold=0.002, SiftExtraction__estimate_affine_shape=1, SiftExtraction__domain_size_pooling=1)
	run_colmap(
		args.colmap_path, log_path, 'sequential_matcher', database_path=database_path, FeatureMatching__use_gpu=0, FeatureMatching__guided_matching=1,
		SequentialMatching__overlap=20, SequentialMatching__quadratic_overlap=0)
	run_colmap(args.colmap_path, log_path, 'mapper', database_path=database_path, image_path=frames_dir, output_path=sparse_dir)

	model_dir = largest_sparse_model(sparse_dir)
	run_colmap(args.colmap_path, log_path, 'model_converter', input_path=model_dir, output_path=cameras_dir, output_type='TXT')
	run_colmap(args.colmap_path, log_path, 'image_undistorter', image_path=frames_dir, input_path=model_dir, output_path=dense_dir, max_image_size=args.max_image_size)
	run_colmap(
		args.colmap_path, log_path, 'patch_match_stereo', workspace_path=dense_dir, PatchMatchStereo__gpu_index=args.device,
		PatchMatchStereo__window_radius=4, PatchMatchStereo__window_step=2, PatchMatchStereo__num_samples=10, PatchMatchStereo__num_iterations=4)
	run_colmap(args.colmap_path, log_path, 'stereo_fusion', workspace_path=dense_dir, output_path=fused_path)


def load_cameras(cameras_dir):
	with open(os.path.join(cameras_dir, 'images.txt')) as images_file:
		image_lines = [line for line in images_file if not line.startswith('#')][0::2]

	image_lines.sort(key=lambda image_line: image_line.split()[9])
	poses = np.array([image_line.split()[1:8] for image_line in image_lines], dtype=np.float64)
	rotations = Rotation.from_quat(poses[:, [1, 2, 3, 0]]).as_matrix()
	positions = -np.einsum('nji,nj->ni', rotations, poses[:, 4:])
	return (positions, rotations)


def camera_up_direction(rotations):
	(_, _, axes) = np.linalg.svd(rotations[:, 0])
	return axes[2] if axes[2] @ rotations[:, 1].mean(axis=0) < 0 else -axes[2]


def camera_focus_point(positions, rotations):
	view_directions = rotations[:, 2]
	projections = np.eye(3) - view_directions[:, :, None] * view_directions[:, None, :]
	return np.linalg.solve(projections.sum(axis=0), (projections @ positions[:, :, None]).sum(axis=0)[:, 0])


def robust_size(points):
	(low, high) = np.percentile(points, [1, 99], axis=0)
	return np.linalg.norm(high - low)


def fit_floor_plane(points, tolerance, max_angle, num_iterations=2000, max_points=100000):
	subset = points[np.random.choice(points.shape[0], min(points.shape[0], max_points), replace=False)]
	triples = subset[np.random.randint(subset.shape[0], size=(num_iterations, 3))]
	plane_normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
	plane_normals = plane_normals / np.maximum(np.linalg.norm(plane_normals, axis=1, keepdims=True), 1e-12)
	plane_normals[plane_normals @ UP_AXIS < 0] *= -1
	upright = plane_normals @ UP_AXIS > math.cos(math.radians(max_angle))

	best_score = -math.inf
	best_plane = None

	for (origin, normal) in zip(triples[upright, 0], plane_normals[upright]):
		heights = (subset - origin) @ normal
		score = np.count_nonzero(np.abs(heights) < tolerance) - 5 * np.count_nonzero(heights < -tolerance)

		if score > best_score:
			best_score = score
			best_plane = (origin, normal)

	(origin, normal) = best_plane
	inliers = points[np.abs((points - origin) @ normal) < tolerance]
	origin = inliers.mean(axis=0)
	(_, _, axes) = np.linalg.svd(inliers - origin, full_matrices=False)
	normal = axes[2] if axes[2] @ UP_AXIS > 0 else -axes[2]
	thickness = 4 * 1.4826 * np.median(np.abs((inliers - origin) @ normal))

	return (origin, normal, thickness)


def rotation_between(source, target):
	axis = np.cross(source, target)
	sine = np.linalg.norm(axis)
	cosine = source @ target

	if sine < 1e-12:
		return np.eye(3)

	axis = axis / sine
	skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
	return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


def yaw_rotation(angle):
	(cosine, sine) = (math.cos(angle), math.sin(angle))
	return np.array([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]])


def outlier_mask(points, num_neighbors, std_ratio):
	(distances, _) = cKDTree(points).query(points, k=num_neighbors + 1, workers=-1)
	spacing = distances[:, 1:].mean(axis=1)
	return spacing < spacing.mean() + std_ratio * spacing.std()


def cluster_mask(points, cell_size, min_cluster_ratio):
	(voxels, voxel_index) = np.unique(np.floor(points / cell_size).astype(np.int64), axis=0, return_inverse=True)
	pairs = cKDTree(voxels).query_pairs(1.8, output_type='ndarray')
	graph = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(voxels.shape[0], voxels.shape[0]))
	(_, voxel_labels) = connected_components(graph, directed=False)
	point_labels = voxel_labels[voxel_index.reshape(-1)]
	cluster_sizes = np.bincount(point_labels)
	focus_distances = np.linalg.norm(points[:, [0, 2]], axis=1)
	focus_distances[cluster_sizes[point_labels] < min_cluster_ratio * cluster_sizes.max()] = math.inf
	return point_labels == point_labels[np.argmin(focus_distances)]


def footprint_angle(points):
	footprint = points[:, [0, 2]]
	hull = footprint[ConvexHull(footprint).vertices]
	edges = np.roll(hull, -1, axis=0) - hull
	best_area = math.inf
	best_angle = 0.0

	for angle in np.arctan2(edges[:, 1], edges[:, 0]):
		(cosine, sine) = (math.cos(-angle), math.sin(-angle))
		rotated = hull @ np.array([[cosine, sine], [-sine, cosine]])
		extents = rotated.max(axis=0) - rotated.min(axis=0)

		if extents[0] * extents[1] < best_area:
			best_area = extents[0] * extents[1]
			best_angle = -angle

	return best_angle


def scale_points_to_unit_sphere(points):
	points = points - (points.min(axis=0) + points.max(axis=0)) / 2
	return points / np.linalg.norm(points, axis=1).max()


def voxel_downsample(voxel_size, points, normals):
	(_, voxel_index, voxel_counts) = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
	voxel_index = voxel_index.reshape(-1)
	attributes = np.concatenate((points, normals), axis=1)
	sums = [np.bincount(voxel_index, weights=attributes[:, column], minlength=voxel_counts.shape[0]) for column in range(attributes.shape[1])]
	means = np.stack(sums, axis=1) / voxel_counts[:, None]
	return (means[:, :3], means[:, 3:])


def point_areas(tree, points, num_neighbors=8):
	(distances, _) = tree.query(points, k=num_neighbors + 1, workers=-1)
	return math.pi * distances[:, -1] ** 2 / num_neighbors


def point_cloud_sdf(points, normals, areas, tree, query_points):
	(distances, _) = tree.query(query_points, workers=-1)
	winding_numbers = pcu.point_cloud_fast_winding_number(points, normals, areas, query_points)
	distances[(winding_numbers > 0.5) & (query_points[:, 1] > points[:, 1].min())] *= -1
	return distances


def save_sample(output_dir, folder, name, points, distances=None):
	sample = points if distances is None else np.concatenate((points, distances[:, None]), axis=1)
	np.save(os.path.join(output_dir, folder, name), sample.astype(np.float32))


def main():
	args = options()
	name = os.path.splitext(os.path.basename(args.input_file))[0]
	colmap_dir = os.path.join(os.path.normpath(args.output_dir) + COLMAP_FOLDER_SUFFIX, name)
	cameras_dir = os.path.join(colmap_dir, 'cameras')
	fused_path = os.path.join(colmap_dir, 'dense', 'fused.ply')

	if not os.path.isfile(fused_path):
		reconstruct_point_cloud(args, colmap_dir, cameras_dir, fused_path)

	np.random.seed(0)

	(camera_positions, camera_rotations) = load_cameras(cameras_dir)
	(points, normals) = pcu.load_mesh_vn(fused_path)
	(points, normals) = (points.astype(np.float64), normals.astype(np.float64))
	print(f'Loaded {points.shape[0]} points reconstructed from {camera_positions.shape[0]} of {len(os.listdir(os.path.join(colmap_dir, "frames")))} images')

	focus_point = camera_focus_point(camera_positions, camera_rotations)
	camera_distance = np.median(np.linalg.norm(camera_positions - focus_point, axis=1))
	rotation = rotation_between(camera_up_direction(camera_rotations), UP_AXIS)
	(points, normals) = ((points - focus_point) @ rotation.T / camera_distance, normals @ rotation.T)
	first_camera = (camera_positions[0] - focus_point) @ rotation.T
	keep = np.linalg.norm(points, axis=1) < args.crop_radius
	(points, normals) = (points[keep], normals[keep])
	print(f'Cropped around the object: {points.shape[0]} points left')

	(floor_origin, floor_normal, floor_thickness) = fit_floor_plane(points, args.floor_tolerance * robust_size(points), args.floor_max_angle)
	keep = (points - floor_origin) @ floor_normal > floor_thickness
	rotation = rotation_between(floor_normal, UP_AXIS)
	(points, normals) = (points[keep] @ rotation.T, normals[keep] @ rotation.T)
	first_camera = first_camera @ rotation.T
	print(f'Removed floor: {points.shape[0]} points left')

	keep = outlier_mask(points, args.outlier_neighbors, args.outlier_std)
	(points, normals) = (points[keep], normals[keep])
	keep = cluster_mask(points, args.cluster_cell_size * robust_size(points), args.min_cluster_ratio)
	(points, normals) = (points[keep], normals[keep])
	print(f'Removed outliers: {points.shape[0]} points left')

	rotation = yaw_rotation(-math.pi / 2 - math.atan2(first_camera[2], first_camera[0]) + math.radians(args.yaw))
	(points, normals) = (points @ rotation.T, normals @ rotation.T)
	rotation = yaw_rotation((footprint_angle(points) + math.pi / 4) % (math.pi / 2) - math.pi / 4)
	(points, normals) = (points @ rotation.T, normals @ rotation.T)
	points = scale_points_to_unit_sphere(points)

	(points, normals) = voxel_downsample(args.voxel_size, points, normals)
	normal_lengths = np.linalg.norm(normals, axis=1)
	keep = normal_lengths > 0.5
	(points, normals) = (points[keep], normals[keep] / normal_lengths[keep, None])
	print(f'Resampled to {points.shape[0]} evenly spaced points')

	tree = cKDTree(points)
	areas = point_areas(tree, points)

	near_surface_points = points[np.random.randint(points.shape[0], size=args.num_sdf_samples)]
	near_surface_points = near_surface_points + np.random.randn(args.num_sdf_samples, 3) * args.sample_dist
	near_surface_distances = point_cloud_sdf(points, normals, areas, tree, near_surface_points)
	uniform_points = sample_uniform_points_sphere(args.num_sdf_samples).numpy().astype(np.float64)
	uniform_distances = point_cloud_sdf(points, normals, areas, tree, uniform_points)
	surface_points = points[np.random.randint(points.shape[0], size=args.num_surface_samples)]

	dataset_dir = os.path.join(args.output_dir, name)
	init_dataset(dataset_dir, True, args)
	save_sample(dataset_dir, UNIFORM_FOLDER, name, uniform_points, uniform_distances)
	save_sample(dataset_dir, NEAR_SURFACE_FOLDER, name, near_surface_points, near_surface_distances)
	save_sample(dataset_dir, SURFACE_FOLDER, name, surface_points)
	save_list(os.path.join(dataset_dir, SAMPLE_LIST_FILE), [name + '.npy'])

	print(f'Inside samples: {100 * np.mean(near_surface_distances < 0):.1f}% near-surface, {100 * np.mean(uniform_distances < 0):.1f}% uniform')
	print(f'Dataset saved to:\n{os.path.abspath(dataset_dir)}')


if __name__ == '__main__':
	main()
