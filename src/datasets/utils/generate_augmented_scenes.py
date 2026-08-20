import os
import numpy as np
import random


from  numpy.typing import NDArray

import open3d as o3d

from scipy.spatial.transform import Rotation  # type: ignore

from open3d.visualization.tensorboard_plugin import summary  # noqa: F401
from torch.utils.tensorboard.writer import SummaryWriter

def vis_ptcloud_with_masks(writer: SummaryWriter, step: int, ptcloud:NDArray[np.float32], masks:NDArray[np.int8]):

    # Find unique instance labels
    # unique_labels = np.unique(masks)
    aggregated_colors = np.zeros((ptcloud.shape[0], 3))

    # Loop through the unique labels
    for midx in range(masks.shape[0]):
        # Index the point cloud to find points belonging to the instance object
        indices = masks[midx,:]==1
        instance_points = ptcloud[indices]

        # Assign a random unique color to the points
        color = np.random.uniform(0, 1, size=(1, 3))

        # Create an Open3D color map for the instance points
        colors = np.tile(color, (instance_points.shape[0], 1))
        colors = o3d.utility.Vector3dVector(colors)

        # Assign the colors to the instance points in the point cloud
        aggregated_colors[indices] = color

    pcd_dict = {
            'vertex_positions': ptcloud,
            'vertex_colors': aggregated_colors,
        }
    writer.add_3d('pcd', pcd_dict, step=step) # type: ignore

def get_random_rotation_matrix() -> NDArray[np.float32]:
    """Generate a random 3D rotation matrix."""
    theta_x = np.random.uniform(0, 2*np.pi)
    theta_y = np.random.uniform(0, 2*np.pi)
    theta_z = np.random.uniform(0, 2*np.pi)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# def overlapping_translation(existing_points:NDArray[np.float32], new_point_cloud: NDArray[np.float32], min_overlap = 0.05) -> NDArray[np.float32]:
#     """
#     Generate a translation vector that ensures the new point cloud overlaps with the existing points.
#     :param existing_points: numpy array of existing points in the scene
#     :param new_point_cloud: numpy array of the new point cloud to be added
#     :param min_overlap: minimum amount of overlap required
#     :return: translation vector
#     """
#     min_coord = np.array([0, 0, 0])  # Define minimum coordinates
#     max_coord = np.array([1.5, 1.5, 1.5])  # Define maximum coordinates

#     if existing_points.size == 0:
#         # If there are no existing points, return a random translation within the defined range
#         return np.random.rand(1, 3) * (max_coord - min_coord) + min_coord

#     # Calculate the bounding box of the existing points
#     existing_min = np.min(existing_points, axis=0)
#     existing_max = np.max(existing_points, axis=0)

#     # Calculate the bounding box of the new point cloud
#     new_min = np.min(new_point_cloud, axis=0)
#     new_max = np.max(new_point_cloud, axis=0)

#     # Determine the overlap region
#     overlap_min = existing_max - new_max - min_overlap
#     overlap_max = existing_min - new_min + min_overlap

#     # Ensure the overlap region is valid
#     overlap_min = np.maximum(overlap_min, min_coord)
#     overlap_max = np.minimum(overlap_max, max_coord)

#     # Generate a random translation vector within the overlap region
#     translation = np.random.rand(1, 3) * (overlap_max - overlap_min) + overlap_min

#     return translation

def farthest_point_sample(point, npoint) -> NDArray[np.float32]:
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def compute_bounding_cylinder(pcd):
    # Calculate centroid
    centroid = np.mean(pcd, axis=0)

    # Compute maximum distance from centroid
    distances = np.linalg.norm(pcd - centroid, axis=1)
    radius = np.max(distances)

    # Determine height (adjust as needed)
    min_z = np.min(pcd[:, 2])
    max_z = np.max(pcd[:, 2])
    height = max_z - min_z

    # Create bounding cylinder (center, radius, height)
    # bounding_cylinder = (centroid, radius, height)

    return {
        'center': centroid,
        'radius': radius,
        'height': height,
    }

def get_trans_rot(pts1, pts2):
    box1 = compute_bounding_cylinder(pts1)
    box2 = compute_bounding_cylinder(pts2)
    return calculate_overlap_translation_rotation(box1, box2)

# def calculate_overlap_translation_rotation(box1, box2, desired_overlap_factor=0.5):
#     """
#     Calculates the translation and rotation vectors to overlap box2 with box1.

#     Args:
#         box1 (dict): Dictionary containing information about the first box.
#             Example: {'center': np.array([x1, y1, z1]), 'radius': r1, 'height': h1}
#         box2 (dict): Dictionary containing information about the second box.
#             Example: {'center': np.array([x2, y2, z2]), 'radius': r2, 'height': h2}
#         desired_overlap_factor (float): Desired overlap factor (between 0 and 1).

#     Returns:
#         tuple: Translation vector (tx, ty, tz) and rotation maxtrix (rx, ry, rz) in radians.
#     """
#     # Calculate the distance between box centers
#     center_distance = np.linalg.norm(box1['center'] - box2['center'])
#     print(f'center_distance {center_distance}')
#     # Calculate the desired overlap distance
#     overlap_distance = (box1['radius'] + box2['radius']) * (1 - desired_overlap_factor)
#     print(f'overlap distnace {overlap_distance}')
#     # Ensure the distance is within bounds
#     # overlap_distance = np.clip(overlap_distance, 0.5, 0.8)
#     print(f'overlap distance {overlap_distance}')

#     # Calculate the translation vector
#     translation_vector = (box1['center'] - box2['center']) * (overlap_distance / center_distance)

#     angle = np.arccos(np.clip(np.dot(box1['center'] / np.linalg.norm(box1['center']), box2['center'] / np.linalg.norm(box2['center'])), -1.0, 1.0))  
#     rotation_angles = (0, 0, angle) 
#     print(f'angle {rotation_angles}')

#     # Adjust for height difference
#     height_difference = box1['height'] - box2['height']
#     print(f'height difference {height_difference}')
#     translation_vector[2] += height_difference / 2

#     return translation_vector, Rotation.from_euler('xyz', rotation_angles).as_matrix()

def random_rotation_matrix():
    """
    Generates a random rotation matrix.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    angle_upright = np.random.uniform(-np.pi / 64, np.pi / 64)
    angle_other = np.random.uniform(-np.pi / 64, np.pi / 64)
    rotation_upright = Rotation.from_euler('z', angle_upright).as_matrix()
    rotation_other = Rotation.from_euler('x', angle_other).as_matrix()
    rotation_matrix = np.dot(rotation_upright, rotation_other)
    return rotation_matrix

def calculate_overlap_translation_rotation(box1, box2, desired_overlap_factor=0.2):
    """
    Calculates the translation and rotation vectors to overlap box2 with box1.

    Args:
        box1 (dict): Dictionary containing information about the first box.
            Example: {'center': np.array([x1, y1, z1]), 'radius': r1, 'height': h1}
        box2 (dict): Dictionary containing information about the second box.
            Example: {'center': np.array([x2, y2, z2]), 'radius': r2, 'height': h2}
        desired_overlap_factor (float): Desired overlap factor (between 0 and 1).

    Returns:
        tuple: Translation vector (tx, ty, tz) and rotation maxtrix (rx, ry, rz) in radians.
    """
    # Calculate the distance between box centers
    # center_distance = np.linalg.norm(box1['center'] - box2['center'])
    # print(f'center_distance {center_distance}')
    # Calculate the desired overlap distance
    # overlap_distance = (box1['radius'] + box2['radius']) * (1 - desired_overlap_factor)
    # print(f'overlap distnace {overlap_distance}')
    # Ensure the distance is within bounds
    # overlap_distance = np.clip(overlap_distance, 0.5, 0.8)
    # print(f'overlap distance {overlap_distance}')

    # Calculate the translation vector
    translation_vector = (box1['center'] - box2['center'])
    # print(translation_vector)
    # angle = np.arccos(np.clip(np.dot(box1['center'] / np.linalg.norm(box1['center']), box2['center'] / np.linalg.norm(box2['center'])), 0.5, 1.5))  
    # rotation_angles = (0, 0, angle) 
    # print(f'angle {rotation_angles}')

    # # Adjust for height difference
    height_difference = box1['height'] - box2['height']
    # # print(f'height difference {height_difference}')
    translation_vector[2] += height_difference / 2

    return translation_vector

def metric_std(A, B):
        
    # standard deviations 
    A_std  = np.std(A, axis=0)
    B_std  = np.std(B, axis=0)
    pooled_std = np.std(np.vstack([A, B]), axis=0)
    
    # metric: ratio of higher individual to pooled std    
    std_ratio = np.max([A_std, B_std], axis=0) / pooled_std
        
    return np.mean(std_ratio)


def random_scale_vector():
    """
    Generates a random scale vector.

    Returns:
        np.ndarray: A 3D scale vector.
    """
    # Generate random scaling factors
    scale_factor = np.random.uniform(0.9, 1.1, size=3)

    return scale_factor

if __name__ == '__main__':
    # print(np.load('/media/data4/maral/Segment_PCD/data/augmented_modelnet_scenes/train/mask_0.npy').dtype)
    # Define the dataset directory
    dataset_dir = '/media/data4/maral/Segment_PCD/data/ModelNet/modelNet40/modelnet40_normal_resampled'
    TOTAL_CATEGORIES = 40  # Total number of categories
    P = 20  # Number of permutations per category
    C = 2  # Number of unique categories in each permutation
    M = 1  # Number of instances per category


    # Step 1: List all object categories
    categories = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    instances = np.loadtxt('/media/data4/maral/Segment_PCD/data/ModelNet/modelNet40/modelnet40_test.txt', dtype=str)
    # print(instances.shape)
    # Initialize storage for all scenes
    # all_scenes = []
    # all_labels = []
    # all_masks = []
    # writer = SummaryWriter(log_dir="/media/data4/maral/Segment_PCD/experiments/ds_viz/modelnetmix_test")

    # Generate S permutations of N unique categories for each of the 40 categories
    for sm in range(TOTAL_CATEGORIES * P):
        print(sm)
        # Step 2: Randomly select N unique categories
        selected_categories = random.sample(categories, C)
        # Step 3: Randomly select M objects from each category

        selected_objects = {category: random.sample([elem for elem in instances if elem.startswith(category)], M) for category in selected_categories}
        # Now you can iterate over the selected objects to create different scenes
        scene_points = []
        labels = []
        for category, objects in selected_objects.items():
            for obj in objects:
                # Load point cloud
                point_cloud = np.loadtxt(os.path.join(dataset_dir, category, f'{obj}.txt'), delimiter=',').astype(np.float32)[:,:3]
                point_cloud = pc_normalize(point_cloud)
                point_cloud = farthest_point_sample(point_cloud,2048)

                r = random_rotation_matrix()
                point_cloud = np.dot(point_cloud, r.T)
                s = random_scale_vector()
                point_cloud = point_cloud * s

                if len(scene_points) > 0:
                #     # print(scene_points[0].shape)
                #     # print(point_cloud.shape)
                    existing_points = np.concatenate(scene_points, axis=0)
                    trans= get_trans_rot(existing_points, point_cloud)
                #     # print(rot.shape)
                    point_cloud += trans
                #     point_cloud = np.dot(point_cloud, rot.T)
                #     print(f'overlap {metric_std(existing_points, point_cloud)}')
                #     print()
                    # print(point_cloud.shape)
                # Add to scene
                scene_points.append(point_cloud)
                # Generate labels
                # labels.append(np.full((point_cloud.shape[0],), category))

        # Combine points and labels for the current scene
        scene = np.concatenate(scene_points, axis=0, dtype=np.float32)
        # label = np.concatenate(labels, axis=0)
        # Create a mask for the current scene
        masks = np.zeros((M * C, scene.shape[0]), dtype=np.int8)
        for i, pcd in enumerate(scene_points):
            start_idx = sum(len(pc) for pc in scene_points[:i])
            end_idx = start_idx + len(pcd)
            masks[i, start_idx:end_idx] = 1

        np.save(f'/media/data4/maral/Segment_PCD/data/augmented_modelnet/test/scenes/scene_{sm}.npy', scene)
        np.save(f'/media/data4/maral/Segment_PCD/data/augmented_modelnet/test/masks/mask_{sm}.npy', masks)
        # vis_ptcloud_with_instances(writer, sm, scene, masks)
        # Add the created scene to the list of all scenes
        # all_scenes.append(scene)
        # all_labels.append(label)
        # all_masks.append(masks)


    # for i, (scene, mask) in tqdm(enumerate(zip(all_scenes, all_masks))):
    #     np.save(f'/media/data4/maral/Segment_PCD/data/augmented_modelnet_scenes/train/scene_{i}.npy', scene)
    #     np.save(f'/media/data4/maral/Segment_PCD/data/augmented_modelnet_scenes/train/mask_{i}.npy', mask)
    #     vis_ptcloud_with_instances(writer, i, scene, mask) 
