import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(input_path='data/raw/sales_data.csv', output_dir='data/processed'):
    """
    Tiền xử lý dữ liệu: Làm sạch, mã hóa biến phân loại và chia tập dữ liệu Train/Test.
    Lưu kết quả vào thư mục data/processed/
    """
    # 1. Xác định đường dẫn tương đối chuẩn của dự án
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_script_dir)            # ML_Lab2/
    
    file_path = os.path.join(project_root, input_path)
    target_dir = os.path.join(project_root, output_dir)
    
    # Tạo thư mục output_dir (data/processed/) nếu chưa tồn tại
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {file_path}")
        print("💡 Hãy chạy 'python3 src/data_collection.py' trước!")
        return

    print("🔄 Đang đọc và tiến hành tiền xử lý dữ liệu...")
    df = pd.read_csv(file_path)

    # 2. Tính biến mục tiêu doanh số (sales) nếu chưa có
    if 'sales' not in df.columns:
        df['sales'] = df['unit_price'] * df['quantity']

    # 3. Loại bỏ các cột ID không mang giá trị dự đoán
    drop_cols = ['order_id', 'customer_id', 'product_id', 'product_name', 'order_date']
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 4. Mã hóa biến chữ/phân loại (Categorical variables) sang dạng Số bằng One-Hot Encoding
    categorical_cols = ['category', 'customer_gender', 'country']
    df_encoded = pd.get_dummies(df_clean, columns=[col for col in categorical_cols if col in df_clean.columns], drop_first=True)

    print(f"✅ Đã tiền xử lý xong! Kích thước bảng dữ liệu mới: {df_encoded.shape[0]} dòng, {df_encoded.shape[1]} cột")

    # 5. Tách thành tập Features (X) và Target (y)
    X = df_encoded.drop(columns=['sales'])
    y = df_encoded['sales']

    # 6. Chia tập Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Lưu kết quả ra các file dữ liệu sạch vào thư mục data/processed/
    X_train.to_csv(os.path.join(target_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(target_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(target_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(target_dir, 'y_test.csv'), index=False)
    df_encoded.to_csv(os.path.join(target_dir, 'sales_cleaned.csv'), index=False)

    print(f"💾 Đã lưu dữ liệu làm sạch tại thư mục: {target_dir}/")
    print(" - X_train.csv, y_train.csv (80% dữ liệu để huấn luyện)")
    print(" - X_test.csv, y_test.csv   (20% dữ liệu để kiểm thử)")

if __name__ == "__main__":
    preprocess_data()