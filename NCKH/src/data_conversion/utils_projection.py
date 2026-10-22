# src/data_conversion/utils_projection.py

import numpy as np
from pyquaternion import Quaternion
from truckscenes.utils.data_classes import Box

def convert_3d_box_to_2d(ts, ann_token, cam_data_token):
    try:
        ann = ts.get('sample_annotation', ann_token)
        ann_box_3d = Box(
            center=ann['translation'],
            size=ann['size'],
            orientation=Quaternion(ann['rotation'])
        )
        corners_3d_world = ann_box_3d.corners()

        cam_data = ts.get('sample_data', cam_data_token)
        calib_data = ts.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(calib_data['camera_intrinsic'])
        cam_translation = np.array(calib_data['translation'])
        cam_rotation = Quaternion(calib_data['rotation'])

        ego_pose_token = cam_data['ego_pose_token']
        ego_pose = ts.get('ego_pose', ego_pose_token)
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        corners_in_ego_frame = corners_3d_world - ego_translation.reshape(3, 1)
        corners_in_ego_frame = ego_rotation.inverse.rotation_matrix @ corners_in_ego_frame

        corners_in_cam_frame = corners_in_ego_frame - cam_translation.reshape(3, 1)
        corners_in_cam_frame = cam_rotation.inverse.rotation_matrix @ corners_in_cam_frame

        points_3d = corners_in_cam_frame
        depths = points_3d[2, :]
        
        projected_points = cam_intrinsic @ points_3d
        points_2d = projected_points[:2, :] / projected_points[2, :]

        valid_mask = depths > 0
        valid_points_2d = points_2d[:, valid_mask]

        if valid_points_2d.shape[1] < 2:
            return None

        x_min = int(np.min(valid_points_2d[0, :]))
        y_min = int(np.min(valid_points_2d[1, :]))
        x_max = int(np.max(valid_points_2d[0, :]))
        y_max = int(np.max(valid_points_2d[1, :]))

        img_width = cam_data['width']
        img_height = cam_data['height']
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width - 1, x_max)
        y_max = min(img_height - 1, y_max)

        if (x_max - x_min) < 5 or (y_max - y_min) < 5:
            return None
            
        return (x_min, y_min, x_max, y_max)

    except Exception as e:
        print(f"Lỗi khi chiếu 3D->2D: {e}")
        return None

def convert_to_yolo_format(box_2d, img_width, img_height):
    xmin, ymin, xmax, ymax = box_2d
    
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    width_norm = width * dw
    height_norm = height * dh
    
    return (x_center_norm, y_center_norm, width_norm, height_norm)