import pandas as pd
from pathlib import Path

# Doc du lieu
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "sales_data.csv")

print(df.head())
print(df.info())

# Kiem tra du lieu thieu
print("\nDu lieu thieu:")
print(df.isnull().sum())

# Kiem tra du lieu trung lap
print("\nSo dong trung lap:")
print(df.duplicated().sum())

# Chuyen cot order_date sang dang ngay thang
df["order_date"] = pd.to_datetime(df["order_date"])

# Kiem tra lai kieu du lieu
print("\nKieu du lieu sau khi chuyen doi:")
print(df["order_date"].dtype)

# Kiem tra gia tri hong hop le
print("\nGia tri nho hon hoac bang 0:")

print("unit_price:", (df["unit_price"] <=0).sum())
print("quantity:",(df["quantity"] <=0).sum())
print("customer_age:", (df["customer_age"] <=0).sum())

# Xu ly khoang trang trong du lieu chu
print("\nXu ly khoang trang:")

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
    print(col, "da xu ly")

# Luu du lieu sau khi tien xu ly
output_path = BASE_DIR / "data" / "processed_data.csv"
df.to_csv(output_path, index=False)

print("\nDa luu du lieu tai:", output_path)

# Kiem tra khoang trang trong du lieu chu
print("\nKiem tra khoang trang:")

for col in df.select_dtypes(include="object").columns:
    print(col, df[col].str.strip().ne(df[col]).sum())
