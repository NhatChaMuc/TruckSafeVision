# (NCKH 202X) Hệ thống Cảnh báo Điểm mù cho Xe tải sử dụng Computer Vision

Đây là dự án NCKH của nhóm với mục tiêu phát hiện các đối tượng (xe máy, người đi bộ) trong vùng điểm mù của xe tải bằng YOLOv8 và bộ dữ liệu MAN TruckScenes.

## 📦 Dữ liệu (Dataset)
Do dung lượng bộ dữ liệu lớn (~10GB), vui lòng tải file dataset từ Google Drive:
👉 [Link Tải Dataset NCKH trên Google Drive](https://drive.google.com/file/d/1TyMA8AM_8QbpYxsodgpjuDOM1lFaH4Po/view)

Sau khi tải về, giải nén và đặt vào thư mục `NCKH/data/` để tiến hành huấn luyện/đánh giá.

---

## 1. Cài đặt Môi trường

Dự án này sử dụng Python 3.10+.

**Bước 1: cây thư mục dự án**
SafeTruckVision/
│
├── .gitignore          # File cấu hình để Git bỏ qua các thư mục
├── README.md           # File hướng dẫn tổng quan về project
├── requirements.txt    # Danh sách các thư viện Python cần cài
│
├── data/
│   ├── raw/            # Nơi chứa dataset GỐC (MAN TruckScenes mini 11GB)
│   └── processed/      # Nơi chứa dữ liệu đã CHUYỂN ĐỔI sang định dạng YOLO
│       ├── images/
│       │   ├── train/  # Ảnh cho tập huấn luyện
│       │   └── val/    # Ảnh cho tập kiểm thử
│       ├── labels/
│       │   ├── train/  # Nhãn (file .txt) cho tập huấn luyện
│       │   └── val/    # Nhãn (file .txt) cho tập kiểm thử
│       └── data.yaml   # File config cho YOLOv8 biết đường dẫn và tên các lớp
│
├── notebooks/          # Nơi chứa các file Jupyter Notebook để khám phá, thử nghiệm
│
├── runs/               # Nơi YOLOv8 tự động lưu kết quả (weights, biểu đồ)
│
├── src/                # Nơi chứa TOÀN BỘ mã nguồn Python chính
│   ├── __init__.py
│   │
│   ├── configs/
│   │   └── roi_definitions.json  # File JSON để lưu tọa độ các vùng điểm mù (ROI)
│   │
│   ├── data_conversion/          # Module QUAN TRỌNG: Chuyển đổi MAN -> YOLO
│   │   ├── __init__.py
│   │   ├── main_converter.py     # Script chính để chạy chuyển đổi 3D -> 2D
│   │   └── utils_projection.py   # Các hàm toán học (chiếu, chuẩn hóa)
│   │
│   ├── inference/                # Module chạy demo (sản phẩm)
│   │   ├── __init__.py
│   │   ├── run_demo.py           # Script để chạy model trên video/webcam
│   │   └── roi_analysis.py       # Code logic kiểm tra va chạm với vùng ROI
│   │
│   └── training/                 # Module huấn luyện
│       ├── __init__.py
│       └── train.py              # Script để bắt đầu quá trình train YOLOv8
│
└── venv/
