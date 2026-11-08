import os
import pandas as pd
import joblib

def run_demo():
    """
    Giả lập dự đoán doanh số cho một đơn hàng mới sử dụng mô hình đã huấn luyện.
    """
    # Vì demo.py nằm ở thư mục gốc nên current_dir chính là project_root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Đường dẫn mô hình và file dữ liệu mẫu đã cập nhật theo cấu trúc mới
    model_path = os.path.join(project_root, 'models', 'RandomForest.pkl')
    x_train_fe_path = os.path.join(project_root, 'data', 'processed', 'X_train_fe.csv')

    if not os.path.exists(model_path):
        print("❌ Không tìm thấy file mô hình! Hãy chắc chắn đã chạy 'python3 src/training.py'")
        return

    # 2. Tải mô hình đã lưu từ thư mục models/
    print("🤖 Đang tải mô hình RandomForest từ thư mục models/...")
    model = joblib.load(model_path)

    # 3. Đọc đúng cấu trúc cột đặc trưng từ data/processed/X_train_fe.csv
    if not os.path.exists(x_train_fe_path):
        print(f"❌ Không tìm thấy file cấu trúc cột tại: {x_train_fe_path}")
        print("💡 Hãy chạy 'python3 src/feature_engineering.py' trước!")
        return

    feature_names = pd.read_csv(x_train_fe_path, nrows=1).columns.tolist()

    # 4. Tạo dữ liệu đầu vào mẫu (Một đơn hàng mới)
    print("\n📝 Khởi tạo dữ liệu đơn hàng thử nghiệm:")
    sample_data = {
        'unit_price': 650,
        'quantity': 3,
        'customer_age': 35,
        'is_young': 0,
        'is_senior': 0,
        'is_bulk_order': 1,
        'price_quantity_ratio': 650 / (3 + 1e-5)
    }

    # Đảm bảo khớp đúng khung các cột mà mô hình đã học
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    for key, value in sample_data.items():
        if key in input_df.columns:
            input_df[key] = value

    # Kích hoạt cột phân loại ngẫu nhiên (VD: loại sản phẩm/quốc gia)
    for col in input_df.columns:
        if 'category' in col.lower() or 'country' in col.lower():
            input_df[col] = 1
            break

    # 5. Thực hiện dự đoán
    predicted_sales = model.predict(input_df)[0]

    print("\n" + "="*50)
    print(" 🚀 KẾT QUẢ DỰ ĐOÁN DOANH SỐ THỬ NGHIỆM (DEMO)")
    print("="*50)
    print(f"- Đơn giá sản phẩm : $650")
    print(f"- Số lượng mua     : 3")
    print(f"- Tuổi khách hàng  : 35")
    print("-" * 50)
    print(f"💰 Doanh số dự đoán thu về: ${predicted_sales:,.2f}")
    print("="*50)

if __name__ == "__main__":
    run_demo()