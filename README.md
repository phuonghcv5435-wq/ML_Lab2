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
