import os
import pandas as pd

def collect_data(source_path='sales_data.csv', output_dir='data', output_file='sales_data.csv'):
    """
    Thu thập và lưu trữ tập dữ liệu thô (Raw Data) vào thư mục data/
    """
    # 1. Tạo thư mục data/ nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: {output_dir}")

    destination_path = os.path.join(output_dir, output_file)

    # 2. Đọc và lưu dữ liệu vào thư mục data/
    if os.path.exists(source_path):
        df = pd.read_csv(source_path)
        df.to_csv(destination_path, index=False)
        print(f"✅ Đã thu thập và lưu dữ liệu thành công tại: {destination_path}")
        print(f"📊 Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
    else:
        print(f"❌ Không tìm thấy file nguồn tại: {source_path}")

if __name__ == "__main__":
    collect_data()
