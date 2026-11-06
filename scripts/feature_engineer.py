import os
import pandas as pd

def engineer_features(input_dir='data', output_dir='data'):
    """
    Tạo các đặc trưng mới (Feature Engineering) từ tập dữ liệu đã tiền xử lý.
    """
    # 1. Định vị đường dẫn thư mục gốc
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # scripts/
    project_root = os.path.dirname(current_script_dir)            # ML_Lab2/
    
    data_folder = os.path.join(project_root, input_dir)
    x_train_path = os.path.join(data_folder, 'X_train.csv')
    x_test_path = os.path.join(data_folder, 'X_test.csv')

    # 2. Kiểm tra dữ liệu đầu vào từ Task 4
    if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
        print(f"❌ Không tìm thấy X_train.csv hoặc X_test.csv tại: {data_folder}")
        print("💡 Hãy chạy 'python3 scripts/data_preprocessing.py' trước!")
        return

    print("🚀 Đang tiến hành tạo đặc trưng mới (Feature Engineering)...")
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    # 3. Hàm tạo đặc trưng mới dùng chung cho cả Train và Test
    def add_custom_features(df):
        df_new = df.copy()

        # Feature 1: Phân nhóm độ tuổi khách hàng
        if 'customer_age' in df_new.columns:
            df_new['is_young'] = (df_new['customer_age'] < 30).astype(int)
            df_new['is_senior'] = (df_new['customer_age'] >= 50).astype(int)

        # Feature 2: Mức độ mua sắm số lượng lớn (Bulk buying flag)
        if 'quantity' in df_new.columns:
            df_new['is_bulk_order'] = (df_new['quantity'] >= 3).astype(int)

        # Feature 3: Tương tác giữa Đơn giá và Số lượng
        if 'unit_price' in df_new.columns and 'quantity' in df_new.columns:
            df_new['price_quantity_ratio'] = df_new['unit_price'] / (df_new['quantity'] + 1e-5)

        return df_new

    # 4. Áp dụng tạo đặc trưng
    X_train_fe = add_custom_features(X_train)
    X_test_fe = add_custom_features(X_test)

    print(f"✅ Đã tạo thêm đặc trưng thành công!")
    print(f" - Số cột ban đầu: {X_train.shape[1]}")
    print(f" - Số cột sau khi Feature Engineering: {X_train_fe.shape[1]}")

    # 5. Lưu đè hoặc xuất file dữ liệu mới cho bước huấn luyện mô hình (Task 6)
    X_train_fe.to_csv(os.path.join(data_folder, 'X_train_fe.csv'), index=False)
    X_test_fe.to_csv(os.path.join(data_folder, 'X_test_fe.csv'), index=False)

    print(f"💾 Đã lưu tập dữ liệu nâng cao thành công:")
    print(" - data/X_train_fe.csv")
    print(" - data/X_test_fe.csv")

if __name__ == "__main__":
    engineer_features()