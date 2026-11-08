# 🛒 Project: Product Sales Prediction (MLOps Architecture)

Dự án bài tập lớn nhóm: Dự đoán doanh số bán hàng dựa trên các thuộc tính sản phẩm và thông tin khách hàng.

## 📁 Cấu trúc thư mục
- `data/`: Chứa dữ liệu thô (`sales_data.csv`) và các tập dữ liệu đã làm sạch (`X_train_fe.csv`, `y_train.csv`,...).
- `doc/`: Chứa các biểu đồ phân tích EDA (.png).
- `model/algorithms/`: Chứa các module định nghĩa lớp mô hình (Linear Regression, Decision Tree, Random Forest, Gradient Boosting).
- `model/`: Chứa các file mô hình nhị phân đã huấn luyện (.pkl).
- `scripts/`: Chứa toàn bộ các script thực thi pipeline dữ liệu và mô hình.

## 🚀 Hướng dẫn chạy quy trình (Pipeline)
1. **Thu thập dữ liệu:** `python3 scripts/data_collection.py`
2. **Phân tích EDA:** `python3 scripts/eda.py`
3. **Tiền xử lý dữ liệu:** `python3 scripts/data_preprocessing.py`
4. **Tạo đặc trưng:** `python3 scripts/feature_engineer.py`
5. **Huấn luyện mô hình:** `python3 scripts/training.py`
6. **Đánh giá mô hình:** `python3 scripts/eval.py`
7. **Chạy thử nghiệm Demo:** `python3 scripts/demo.py`




ML_Lab2/
├── data/                          # Quản lý dữ liệu
│   ├── raw/                       # sales_data.csv
│   └── processed/                 # sales_cleaned.csv, X_train.csv, y_train.csv,...
│
├── docs/                          # Tài liệu & Báo cáo
│   ├── description.md             # Mô tả đề tài / bài lab
│   └── report.md                  # Báo cáo kết quả
│
├── figures/                       # Hình ảnh biểu đồ xuất ra từ EDA (tránh để ngoài gốc)
│   ├── eda_correlation_heatmap.png
│   ├── eda_sales_by_category.png
│   ├── eda_sales_by_country.png
│   └── eda_sales_distribution.png
│
├── models/                        # Chứa các file mô hình binary đã train (KHÔNG để code ở đây)
│   ├── DecisionTree.pkl
│   ├── GradientBoosting.pkl
│   ├── LinearRegression.pkl
│   └── RandomForest.pkl
│
├── src/                           # Toàn bộ mã nguồn Python (.py)
│   ├── __init__.py
│   ├── algorithms/                # Các thuật toán / lớp mô hình tự định nghĩa
│   │   ├── decision_tree.py
│   │   ├── gradient_boosting.py
│   │   ├── linear_regression.py
│   │   └── random_forest.py
│   ├── data_collection.py         # Thu thập dữ liệu
│   ├── data_preprocessing.py      # Làm sạch dữ liệu
│   ├── feature_engineering.py     # Tạo/biến đổi tính năng
│   ├── eda.py                     # Vẽ biểu đồ & khám phá dữ liệu
│   ├── training.py                # Script huấn luyện
│   └── eval.py                    # Đánh giá mô hình
│
├── demo.py                        # Script dùng để chạy demo nhanh (nếu có)
├── .gitignore                     # Bỏ qua __pycache__, models/*.pkl,...
├── README.md                      # Hướng dẫn tổng quan bài Lab
└── requirements.txt               # Thư viện sử dụng