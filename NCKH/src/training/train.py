# src/training/train.py

import os
from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt') 

    script_dir = os.path.dirname(__file__)
    data_yaml_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'processed', 'data.yaml'))

    print(f"Bắt đầu huấn luyện với file config: {data_yaml_path}")

    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        device='cpu',
        batch=16,
        name='yolov8s_safetruck_mini'
    )

    print("--- HUẤN LUYỆN HOÀN TẤT ---")
    print(f"Model tốt nhất được lưu tại: {results.save_dir}")

if __name__ == '__main__':
    main()