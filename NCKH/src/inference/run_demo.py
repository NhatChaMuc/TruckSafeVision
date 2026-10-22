# src/inference/run_demo.py
import cv2
from ultralytics import YOLO
import os
import time

from roi_analysis import check_collision, draw_overlays, is_within_distance

# --- ĐỊNH NGHĨA MAPPING LỚP (COCO ID -> Tên của bạn) ---
# COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
CUSTOM_CLASSES = {
    0: "PERSON",
    1: "BICYCLE",
    2: "CAR",       # Bao gồm cả xe bán tải nhỏ, sedan, SUV
    3: "MOTO",
    5: "TRUCK",     # Gom xe buýt vào nhóm xe lớn
    7: "TRUCK"      # Xe tải
}

def main():
    script_dir = os.path.dirname(__file__)
    weights_path = os.path.abspath(os.path.join(
        script_dir, '..', 'runs', 'detect', 'yolov8s_safetruck_mini', 'weights', 'best_model_safetruck.pt'
    ))

    # Fallback model
    if not os.path.exists(weights_path):
        weights_path = 'yolov8s.pt'
        
    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    source = './videos/my_test.mp4' 
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Lỗi mở video: {source}")
        return

    # Full màn hình
    cv2.namedWindow("SafeTruckVision", cv2.WINDOW_NORMAL)

    print("--- STARTING ---")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
    
        # 1. Vẽ vùng ROI & Đường giới hạn khoảng cách
        frame = draw_overlays(frame)

        # 2. Predict
        # Lưu ý: Chỉ detect các class mình quan tâm để nhẹ máy và đỡ nhiễu
        # classes=[0, 1, 2, 3, 5, 7] tương ứng person, bike, car, moto, bus, truck
        results = model.predict(frame, device='cpu', conf=0.35, iou=0.5, classes=[0, 1, 2, 3, 5, 7], verbose=False)

        for res in results:
            boxes = res.boxes
            for box in boxes:
                # Toạ độ
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # --- KIỂM TRA KHOẢNG CÁCH ---
                # Nếu chân vật thể (y2) nằm trên vạch giới hạn -> Bỏ qua
                if not is_within_distance(y2):
                    continue 

                # Lấy Class ID gốc của COCO
                cls_id_coco = int(box.cls[0].cpu().numpy())
                
                # --- XỬ LÝ PHÂN LOẠI (MAPPING) ---
                if cls_id_coco in CUSTOM_CLASSES:
                    class_name = CUSTOM_CLASSES[cls_id_coco]
                else:
                    class_name = "UNKNOWN"

                conf = float(box.conf[0].cpu().numpy())

                # (FIX LỖI MOTO vs BICYCLE)
                # Nếu nhận diện là Xe đạp (1) nhưng độ tin cậy thấp < 0.5 -> Khả năng cao là nhầm -> Bỏ qua
                if cls_id_coco == 1 and conf < 0.5:
                    continue

                # 3. Check logic va chạm
                status, color = check_collision([x1, y1, x2, y2])
                
                # Cảnh báo
                if status == "DANGER":
                    cv2.putText(frame, f"CANH BAO: {class_name}!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                # Vẽ Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label nền màu
                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("SafeTruckVision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()