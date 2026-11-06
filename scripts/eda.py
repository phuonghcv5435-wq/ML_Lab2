import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data_path='data/sales_data.csv', output_dir='doc'):
    """
    Thực hiện Phân tích Khám phá Dữ liệu (EDA) và xuất các biểu đồ phân tích.
    """
    # 1. Xác định đường dẫn tương đối chuẩn của dự án
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # scripts/
    project_root = os.path.dirname(current_script_dir)            # Thư mục gốc dự án
    
    file_path = os.path.join(project_root, data_path)
    save_dir = os.path.join(project_root, output_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Kiểm tra file dữ liệu
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {file_path}")
        print("💡 Hãy chạy 'python3 scripts/data_collection.py' trước!")
        return

    print("📊 Đang đọc dữ liệu để tiến hành EDA...")
    df = pd.read_csv(file_path)

    # Tính biến mục tiêu doanh số (Sales) nếu chưa có
    if 'sales' not in df.columns:
        df['sales'] = df['unit_price'] * df['quantity']

    # 3. In thông tin thống kê tổng quan ra màn hình Terminal
    print("\n" + "="*50)
    print(" 🔍 THÔNG TIN TỔNG QUAN DỮ LIỆU")
    print("="*50)
    print(f"- Số lượng bản ghi (rows): {df.shape[0]}")
    print(f"- Số lượng thuộc tính (cols): {df.shape[1]}")
    print("\n--- Kiểu dữ liệu các cột ---")
    print(df.dtypes)
    
    print("\n--- Kiểm tra dữ liệu thiếu (Missing values) ---")
    print(df.isnull().sum())

    print("\n--- Thống kê mô tả số liệu (Numerical Summary) ---")
    print(df.describe().T)

    # Cấu hình giao diện biểu đồ
    sns.set_theme(style="whitegrid")

    # 4. Vẽ biểu đồ 1: Phân bố Doanh số (Sales Distribution)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['sales'], kde=True, color='teal', bins=20)
    plt.title('Phân bố Doanh số bán hàng (Sales Distribution)', fontsize=14, fontweight='bold')
    plt.xlabel('Doanh số ($)', fontsize=12)
    plt.ylabel('Số lượng đơn hàng', fontsize=12)
    plt.tight_layout()
    plot1_path = os.path.join(save_dir, 'eda_sales_distribution.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"✅ Đã xuất biểu đồ 1: {plot1_path}")

    # 5. Vẽ biểu đồ 2: Tổng Doanh số theo Danh mục Sản phẩm (Category)
    plt.figure(figsize=(8, 5))
    category_sales = df.groupby('category')['sales'].sum().reset_index().sort_values(by='sales', ascending=False)
    sns.barplot(x='category', y='sales', data=category_sales, hue='category', palette='Blues_d', legend=False)
    plt.title('Tổng Doanh số theo Danh mục Sản phẩm', fontsize=14, fontweight='bold')
    plt.xlabel('Danh mục sản phẩm', fontsize=12)
    plt.ylabel('Tổng Doanh số ($)', fontsize=12)
    plt.tight_layout()
    plot2_path = os.path.join(save_dir, 'eda_sales_by_category.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"✅ Đã xuất biểu đồ 2: {plot2_path}")

    # 6. Vẽ biểu đồ 3: Doanh số theo Quốc gia (Country)
    plt.figure(figsize=(10, 5))
    country_sales = df.groupby('country')['sales'].sum().reset_index().sort_values(by='sales', ascending=False)
    sns.barplot(x='country', y='sales', data=country_sales, hue='country', palette='viridis', legend=False)
    plt.title('Tổng Doanh số theo Quốc gia Khách hàng', fontsize=14, fontweight='bold')
    plt.xlabel('Quốc gia', fontsize=12)
    plt.ylabel('Tổng Doanh số ($)', fontsize=12)
    plt.tight_layout()
    plot3_path = os.path.join(save_dir, 'eda_sales_by_country.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"✅ Đã xuất biểu đồ 3: {plot3_path}")

    # 7. Vẽ biểu đồ 4: Ma trận tương quan giữa các biến số (Correlation Heatmap)
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Ma trận tương quan giữa các biến số', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot4_path = os.path.join(save_dir, 'eda_correlation_heatmap.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"✅ Đã xuất biểu đồ 4: {plot4_path}")

    print("\n🎉 Hoàn thành Phân tích EDA! Tất cả biểu đồ đã được lưu tại thư mục doc/")

if __name__ == "__main__":
    perform_eda()