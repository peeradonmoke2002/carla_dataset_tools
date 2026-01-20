#!/usr/bin/python3
import copy
import numpy as np
import open3d as o3d
import cv2
import os
import sys
from pathlib import Path
import transforms3d.euler

sys.path.append(Path(__file__).parent.parent.as_posix())
from core.geometry import Transform, Location
from core.types import ObjectLabel
from core.transform import bbox_to_o3d_bbox


class Param:
    POINTS_MIN_CAR = 20       # Minimum LiDAR points for cars
    POINTS_MIN_PEDESTRIAN = 5  # Minimum LiDAR points for pedestrians (smaller objects)
    POINTS_MIN = 20           # Default (legacy)
    RANGE_MIN = 1.0
    RANGE_MAX = 150.0

    # Range-based filtering (stable approach from CARTI_Dataset)
    # Uses rectangular bounding box instead of point cloud validation
    USE_RANGE_BASED_FILTER = True  # Set to False to use point cloud-based filtering
    RANGE_BOX_SIZE = 51.2  # Detection range in meters (±51.2m in X and Y from sensor)
    
    # Ego vehicle filter - exclude objects too close to sensor (likely ego vehicle)
    EGO_FILTER_DISTANCE = 2.0  # Minimum distance from sensor to include object (meters)
    
    # 360° labeling mode (skip camera filters, label all directions)
    SKIP_CAMERA_FILTERS = True  # Set to True for 360° LiDAR-only labeling


def transform_lidar_point_to_cam(point, lidar_trans: Transform, cam_trans: Transform):
    p = np.append(point, [1.0])
    T_wl = lidar_trans.get_matrix()
    T_cw = cam_trans.get_inverse_matrix()
    T_cl = np.matmul(T_cw, T_wl)
    p_c = np.matmul(T_cl, p)
    return p_c


def project_point_to_image(point_in_cam, cam_mat: np.array):
    """
    Project 3D point in camera coordinate to 2D image plane.

    Args:
        point_in_cam: 3D point in camera coordinate [x, y, z, 1]
        cam_mat: Camera intrinsic matrix (3x3)

    Returns:
        2D pixel coordinates [u, v] as numpy array, or None if point is behind camera
    """
    # Check depth: point must be in front of camera (z > 0)
    if point_in_cam[2] <= 0:
        return None

    p_c = point_in_cam[0:3] / point_in_cam[2]
    p_uv = np.matmul(cam_mat, p_c)
    p_uv = p_uv[0:2].astype(int)

    return p_uv


def transform_o3d_bbox(o3d_bbox: o3d.geometry.OrientedBoundingBox, transform_mat: np.array):
    o3d_bbox.rotate(transform_mat[0:3, 0:3], np.array([0, 0, 0]))
    o3d_bbox.translate(transform_mat[0:3, 3])
    return o3d_bbox


def bbox_to_o3d_bbox_in_target_coordinate(label: ObjectLabel, target_transform: Transform, adjust_pedestrian_z: bool = False):
    """
    Transform object label's bounding box to target coordinate system.
    
    Note on CARLA coordinate handling:
    - For Cars: actor.get_transform().location.z returns the lowest point (ground level)
    - For Pedestrians: actor.get_transform().location.z returns the body center
    
    However, the bounding_box.location already contains the local offset from actor origin
    to the bbox center, so the transformation should handle this correctly.
    The adjust_pedestrian_z flag is kept for backward compatibility but disabled by default
    since the bbox.location already accounts for the proper offset.
    """
    world_to_target = target_transform.get_inverse_matrix()
    label_to_world = label.transform.get_matrix()
    label_in_target = np.matmul(world_to_target, label_to_world)
    o3d_bbox = bbox_to_o3d_bbox(label.bounding_box)
    o3d_bbox = transform_o3d_bbox(o3d_bbox, label_in_target)
    
    # Only apply pedestrian Z adjustment if explicitly requested
    # This is typically not needed since bounding_box.location already has correct offset
    if adjust_pedestrian_z and label.label_type == 'Pedestrian':
        # In LiDAR/world coords: Z is up, so subtract half height to lower the box
        center = np.array(o3d_bbox.center)  # Make a writable copy
        center[2] = center[2] - o3d_bbox.extent[2] / 2.0  # Lower by half height
        o3d_bbox.center = center
    
    o3d_bbox.color = np.array([1.0, 0, 0])
    return o3d_bbox


def o3d_bbox_rotation_to_rpy(o3d_bbox: o3d.geometry.OrientedBoundingBox):
    # o3d bbox R to euler RPY in radian
    roll, pitch, yaw = transforms3d.euler.mat2euler(o3d_bbox.R)
    return roll, pitch, yaw


def cal_truncated(image_length, image_width, bbox_2d: list) -> float:
    # Calculate truncated.
    # from 0 (non-truncated) to 1 (truncated), where
    # truncated refers to the object leaving image boundaries

    # [x_min y_min x_max y_max]
    bbox_2d_in_img = copy.deepcopy(bbox_2d)
    bbox_2d_in_img[0] = max(bbox_2d[0], 0)
    bbox_2d_in_img[1] = max(bbox_2d[1], 0)
    bbox_2d_in_img[2] = min(bbox_2d[2], image_width)
    bbox_2d_in_img[3] = min(bbox_2d[3], image_length)

    size1 = (bbox_2d_in_img[2] - bbox_2d_in_img[0]) * (bbox_2d_in_img[3] - bbox_2d_in_img[1])
    size2 = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])

    # Avoid division by zero - if bbox has zero or negative area, consider it fully truncated
    if size2 <= 0:
        return 1.0

    truncated = size1 / size2
    truncated = max(truncated, 0.0)
    truncated = min(truncated, 1.0)
    truncated = 1.0 - truncated
    return truncated


def cal_occlusion(pcd: o3d.geometry.PointCloud, bbox_3d: o3d.geometry.OrientedBoundingBox, points_min=None):
    """
    Calculate occlusion level based on number of LiDAR points in bounding box.

    Args:
        pcd: Open3D point cloud
        bbox_3d: Open3D oriented bounding box
        points_min: Minimum points threshold. If None, uses Param.POINTS_MIN.
                   Use Param.POINTS_MIN_PEDESTRIAN for pedestrians (smaller threshold).

    Returns:
        occlusion: -1 (invalid/too few points), 0 (fully visible), 1 (partly occluded), 2 (heavily occluded)
    """
    if points_min is None:
        points_min = Param.POINTS_MIN

    occlusion = 2
    p_in_bbox = bbox_3d.get_point_indices_within_bounding_box(pcd.points)
    p_num = len(p_in_bbox)
    if p_num < points_min:
        occlusion = -1
        return occlusion
    elif p_num > points_min:
        occlusion = 0
    if p_num + bbox_3d.center[0] < 250:
        occlusion = 1
    if p_num + bbox_3d.center[0] < 125:
        occlusion = 2
    return occlusion


def is_valid_distance(source_location: Location, target_location: Location):
    dist = np.linalg.norm(source_location.get_vector() - target_location.get_vector())
    if Param.RANGE_MIN < dist < Param.RANGE_MAX:
        return True
    else:
        return False


def is_in_range_box(source_location: Location, target_location: Location, range_box_size=None):
    """
    Check if target is within a rectangular range box around source.

    This is a more stable approach than point cloud-based validation,
    following the CARTI_Dataset methodology. Uses axis-aligned bounding box
    check instead of Euclidean distance.

    Args:
        source_location: Sensor location (e.g., LiDAR position)
        target_location: Object location (e.g., vehicle or pedestrian)
        range_box_size: Half-width of detection box in meters (default: Param.RANGE_BOX_SIZE)

    Returns:
        True if target is within range box, False otherwise

    Example:
        If range_box_size = 51.2, the detection box spans:
        - X: [source.x - 51.2, source.x + 51.2]
        - Y: [source.y - 51.2, source.y + 51.2]
        - Z: no restriction (all heights)
    """
    if range_box_size is None:
        range_box_size = Param.RANGE_BOX_SIZE

    source_vec = source_location.get_vector()
    target_vec = target_location.get_vector()

    # Filter out ego vehicle - objects too close to sensor are likely the ego vehicle
    dist = np.linalg.norm(source_vec[0:2] - target_vec[0:2])  # 2D distance (XY plane)
    if dist < Param.EGO_FILTER_DISTANCE:
        return False

    # Check X and Y within rectangular bounds
    # Z (height) is not restricted - objects at any height within XY range are valid
    dx = abs(target_vec[0] - source_vec[0])
    dy = abs(target_vec[1] - source_vec[1])

    if dx <= range_box_size and dy <= range_box_size:
        return True
    else:
        return False


def write_pointcloud(output_dir: str, frame_id: str, lidar_data: np.array):
    lidar_dir = f"{output_dir}/velodyne"
    os.makedirs(lidar_dir, exist_ok=True)
    file_path = "{}/{}.bin".format(lidar_dir, frame_id)
    lidar_data.tofile(file_path)


def write_image(output_dir: str, frame_id: str, image: np.array):
    image_dir = f"{output_dir}/image_2"
    os.makedirs(image_dir, exist_ok=True)
    file_path = "{}/{}.png".format(image_dir, frame_id)
    cv2.imwrite(file_path, image)


def write_label(output_dir, frame_id, kitti_labels):
    label_dir = f"{output_dir}/label_2"
    os.makedirs(label_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(label_dir, frame_id)

    if len(kitti_labels) < 1:
        kitti_labels.append('DontCare -1 -1 -10 522.25 202.35 547.77 219.71 -1 -1 -1 -1000 -1000 -1000 -10 -10 \n')

    with open(file_path, 'w') as label_file:
        label_file.writelines(kitti_labels)


def write_calib(output_dir, frame_id, lidar_trans: Transform, cam_trans: Transform, camera_mat: np.array):
    """ Saves the calibration matrices to a file.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame.
    """
    calib_dir = f"{output_dir}/calib"
    os.makedirs(calib_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(calib_dir, frame_id)

    camera_mat = np.concatenate((camera_mat, np.array([[0.0], [0.0], [0.0]])), axis=1)
    camera_mat = camera_mat.reshape(1, 12)
    camera_mat_str = ""
    for x in camera_mat[0]:
        camera_mat_str += str(x)
        camera_mat_str += ' '
    camera_mat_str += '\n'

    calib_str = list()
    calib_str.append(f"P0: {camera_mat_str}")
    calib_str.append(f"P1: {camera_mat_str}")
    calib_str.append(f"P2: {camera_mat_str}")
    calib_str.append(f"P3: {camera_mat_str}")

    calib_str.append("R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 \n")

    velo_to_cam = np.matmul(cam_trans.get_inverse_matrix(), lidar_trans.get_matrix())
    velo_to_cam = velo_to_cam[0:3, :]
    velo_to_cam = velo_to_cam.reshape(1, 12).tolist()
    velo_to_cam_str = "Tr_velo_to_cam: "
    for x in velo_to_cam[0]:
        velo_to_cam_str += str(x)
        velo_to_cam_str += ' '
    velo_to_cam_str += '\n'

    calib_str.append(velo_to_cam_str)

    with open(file_path, 'w') as calib_file:
        calib_file.writelines(calib_str)
        calib_file.close()


def generate_kitti_labels(label_type: str,
                          truncated: float,
                          occlusion: float,
                          alpha: float,
                          bbox_2d: list,
                          bbox_3d: o3d.geometry.OrientedBoundingBox,
                          rotation_y: float):
    # Note: Kitti Object 3d bbox location is bottom-center (ground level), not the bbox center
    # This function is for CAMERA coordinate system
    # KITTI camera coords: X=right, Y=down, Z=forward
    # KITTI dimensions order: Height, Width, Length
    label_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n".format(label_type, truncated, occlusion, alpha,
                                                                         bbox_2d[0], bbox_2d[1],
                                                                         bbox_2d[2], bbox_2d[3],
                                                                         bbox_3d.extent[2],
                                                                         bbox_3d.extent[1],
                                                                         bbox_3d.extent[0],
                                                                         bbox_3d.center[0],
                                                                         bbox_3d.center[1] + (bbox_3d.extent[2] / 2.0),
                                                                         bbox_3d.center[2],
                                                                         rotation_y)
    return label_str


# def generate_kitti_labels_lidar(label_type: str,
#                                 truncated: float,
#                                 occlusion: float,
#                                 alpha: float,
#                                 bbox_2d: list,
#                                 bbox_3d: o3d.geometry.OrientedBoundingBox,
#                                 rotation_y: float):
#     # Note: For LiDAR coordinate system, use bottom-center of bbox
#     # This function is for LIDAR coordinate system
#     label_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n".format(label_type, truncated, occlusion, alpha,
#                                                                          bbox_2d[0], bbox_2d[1],
#                                                                          bbox_2d[2], bbox_2d[3],
#                                                                          bbox_3d.extent[2],
#                                                                          bbox_3d.extent[1],
#                                                                          bbox_3d.extent[0],
#                                                                          bbox_3d.center[0],
#                                                                          bbox_3d.center[1] + (bbox_3d.extent[2] / 2.0),
#                                                                          bbox_3d.center[2],
#                                                                          rotation_y)
#     return label_str