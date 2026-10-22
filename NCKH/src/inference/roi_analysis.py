# src/inference/roi_analysis.py
import cv2
import numpy as np

# --- CẤU HÌNH VÙNG (Bạn cần chỉnh lại toạ độ theo video thực tế) ---

# 1. Đường giới hạn khoảng cách (Cắt bỏ những gì ở xa quá 20m)
# Trong ảnh toạ độ (0,0) là góc trên cùng bên trái.
# Giá trị y càng nhỏ là càng ở xa. 
# Ví dụ: Video cao 720px, muốn lấy 20m gần mũi xe thì chỉ lấy từ y=400 trở xuống.
DISTANCE_LIMIT_Y = 265  # <-- CHỈNH SỐ NÀY: Giảm xuống nếu muốn nhìn xa hơn, Tăng lên nếu muốn nhìn gần hơn.

# 2. Vùng Nguy hiểm (Sát xe tải - Đỏ)
ROI_DANGER = np.array([
    [300, 500],  
    [800, 500],  
    [850, 650],  
    [250, 650]   
], np.int32)

# 3. Vùng Cảnh báo (Điểm mù - Vàng)
ROI_WARNING = np.array([
    [100, 400],  
    [1000, 400],  
    [1100, 700],  
    [50, 700]   
], np.int32)

def is_within_distance(box_ymax):
    """Kiểm tra vật thể có nằm trong khoảng cách cho phép không"""
    # Nếu chân vật thể nằm CAO HƠN dòng giới hạn -> Quá xa -> Bỏ qua
    if box_ymax < DISTANCE_LIMIT_Y:
        return False
    return True

def check_collision(detection_box):
    """
    Check va chạm 3 mức độ: DANGER, WARNING, SAFE
    """
    x_min, y_min, x_max, y_max = detection_box
    foot_point = (int((x_min + x_max) / 2), int(y_max))

    # Check vùng Nguy hiểm
    if cv2.pointPolygonTest(ROI_DANGER, foot_point, False) >= 0:
        return "DANGER", (0, 0, 255) # Đỏ

    # Check vùng Cảnh báo
    elif cv2.pointPolygonTest(ROI_WARNING, foot_point, False) >= 0:
        return "WARNING", (0, 255, 255) # Vàng

    return "SAFE", (0, 255, 0) # Xanh lá

def draw_overlays(frame):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Vẽ vùng Cảnh báo (Vàng) & Nguy hiểm (Đỏ)
    cv2.fillPoly(overlay, [ROI_WARNING], (0, 255, 255))
    cv2.fillPoly(overlay, [ROI_DANGER], (0, 0, 255))

    # Vẽ đường giới hạn khoảng cách (Màu xanh dương) để debug
    cv2.line(overlay, (0, DISTANCE_LIMIT_Y), (w, DISTANCE_LIMIT_Y), (255, 0, 0), 2)
    cv2.putText(overlay, "Max Distance (20m limit)", (10, DISTANCE_LIMIT_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)