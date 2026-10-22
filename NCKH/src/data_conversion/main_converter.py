# src/data_conversion/main_converter.py

import os
from pathlib import Path
from tqdm import tqdm
import shutil
from truckscenes.truckscenes import TruckScenes

from utils_projection import convert_3d_box_to_2d, convert_to_yolo_format

DATA_ROOT = Path("../data/raw/man-truckscenes")
VERSION = "v1.0-mini"
OUTPUT_ROOT = Path("../data/processed")
IMAGE_DIR = OUTPUT_ROOT / "images" / "train"
LABEL_DIR = OUTPUT_ROOT / "labels" / "train"

TARGET_CAMERAS = [
    'CAMERA_RIGHT_BACK', 
    'CAMERA_LEFT_BACK', 
    'CAMERA_LEFT_FRONT', 
    'CAMERA_RIGHT_FRONT'
]

TARGET_CLASSES = {
    'human.pedestrian': 0,
    'vehicle.bicycle': 1,
    'vehicle.motorcycle': 2,
    'vehicle.car': 3,
    'vehicle.truck': 4,
    'vehicle.bus': 5,
}
CLASS_MAP = {name: idx for idx, name in enumerate(TARGET_CLASSES.keys())}

def main():
    print(f"Đang tải dataset: {VERSION}...")
    ts = TruckScenes(dataroot=DATA_ROOT, version=VERSION)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Bắt đầu xử lý {len(ts.sample)} samples...")

    for sample in tqdm(ts.sample, desc="Processing Samples"):

        for cam_name in TARGET_CAMERAS:

            if cam_name not in sample['data']:
                continue
            
            cam_data_token = sample['data'][cam_name]
            cam_data = ts.get('sample_data', cam_data_token)

            image_path = DATA_ROOT / VERSION / cam_data['filename']
            if not image_path.exists():
                continue
            
            img_height = cam_data['height']
            img_width = cam_data['width']
            
            yolo_labels = []

            for ann_token in sample['anns']:
                ann = ts.get('sample_annotation', ann_token)

                if ann['category_name'] in TARGET_CLASSES:

                    box_2d = convert_3d_box_to_2d(ts, ann_token, cam_data_token)
                    
                    if box_2d:
                        yolo_box = convert_to_yolo_format(box_2d, img_width, img_height)
                        class_id = CLASS_MAP[ann['category_name']]
                        
                        yolo_line = f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}"
                        yolo_labels.append(yolo_line)

            if yolo_labels:
                output_name = f"{sample['token']}_{cam_name}"
                output_image_path = IMAGE_DIR / f"{output_name}.jpg"
                output_label_path = LABEL_DIR / f"{output_name}.txt"

                shutil.copyfile(image_path, output_image_path)

                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))

    print(f"--- HOÀN TẤT! ---")
    print(f"Dữ liệu đã được xử lý và lưu tại: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()