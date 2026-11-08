import os
import pandas as pd
import numpy as np

def add_custom_features(df):
    """
    Tạo các đặc trưng mới (Feature Engineering) từ tập dữ liệu đã tiền xử lý.
    Dựa trên các yêu cầu: Product Info, Customer Demographics, Seasonality, Marketing.
    """
    # Tạo bản sao để không làm thay đổi dữ liệu gốc
    data = df.copy()
    
    # -------------------------------------------------------------
    # 1. Product Features (Đặc trưng Sản phẩm)
    # -------------------------------------------------------------
    # Tỷ lệ giá so với giá trung bình trong cùng danh mục
    if 'category' in data.columns and 'price' in data.columns:
        category_avg_price = data.groupby('category')['price'].transform('mean')
        data['price_to_cat_avg_ratio'] = data['price'] / (category_avg_price + 1e-5)
    
    # Chỉ số phổ biến (Popularity Score = Reviews * Rating)
    if 'number_of_reviews' in data.columns and 'rating' in data.columns:
        data['popularity_score'] = data['number_of_reviews'] * data['rating']
    
    # -------------------------------------------------------------
    # 2. Customer & Purchase Features (Đặc trưng Khách hàng)
    # -------------------------------------------------------------
    # Giá trị vòng đời khách hàng (Customer Lifetime Value)
    if 'total_spend' in data.columns and 'purchase_frequency' in data.columns:
        data['customer_lifetime_value'] = data['total_spend'] * data['purchase_frequency']

    # -------------------------------------------------------------
    # 3. Time & Seasonality Features (Đặc trưng Thời gian & Mùa vụ)
    # -------------------------------------------------------------
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        # Đánh dấu mùa lễ hội (Tháng 11, 12 là mùa mua sắm cao điểm)
        data['is_holiday_season'] = data['month'].isin([11, 12]).astype(int)

    return data

# SỬA: Thay mặc định input_dir và output_dir từ 'data' sang 'data/processed'
def engineer_features(input_dir='data/processed', output_dir='data/processed'):
    """
    Hàm thực thi chính: Đọc file X_train/X_test từ data/processed/, tạo đặc trưng và lưu X_train_fe/X_test_fe vào data/processed/.
    """
    # 1. Định vị đường dẫn động để tránh lỗi máy khác nhau
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Thư mục src/
    project_root = os.path.dirname(current_script_dir)             # Thư mục gốc dự án
    
    data_folder = os.path.join(project_root, input_dir)
    x_train_path = os.path.join(data_folder, 'X_train.csv')
    x_test_path = os.path.join(data_folder, 'X_test.csv')
    
    # 2. Kiểm tra xem Task tiền xử lý đã chạy ra dữ liệu chưa
    if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
        print(f"❌ LỖI: Không tìm thấy X_train.csv hoặc X_test.csv tại thư mục: {data_folder}")
        print("💡 Hãy chạy file 'src/data_preprocessing.py' trước!")
        return

    print("🚀 Đang tiến hành tạo đặc trưng mới (Feature Engineering)...")
    
    # 3. Nạp dữ liệu
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    
    # 4. Áp dụng Feature Engineering cho cả Train và Test
    X_train_fe = add_custom_features(X_train)
    X_test_fe = add_custom_features(X_test)
    
    # 5. Lưu ra các file đầu ra tại data/processed/ (chữ 'fe' có nghĩa là Feature Engineered)
    out_train_path = os.path.join(project_root, output_dir, 'X_train_fe.csv')
    out_test_path = os.path.join(project_root, output_dir, 'X_test_fe.csv')
    
    X_train_fe.to_csv(out_train_path, index=False)
    X_test_fe.to_csv(out_test_path, index=False)
    
    print(f"✅ Thành công! Đã tạo và lưu các file có đặc trưng mới tại:")
    print(f"   - {out_train_path}")
    print(f"   - {out_test_path}")

if __name__ == "__main__":
    # Khởi chạy hàm chính khi chạy file script này
    engineer_features()