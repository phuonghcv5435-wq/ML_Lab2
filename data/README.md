# Dataset

sales_data.csv is a sourced subset (first 150 transactions) from the public GitHub Gist summerofgeorge/sales_data.csv, described as a simulated sales dataset.

Source: https://gist.github.com/summerofgeorge/33dbb09adbdea670c31ee2c4fc9f2617

The source rows contain product, customer, price, quantity, date, demographics and country. Marketing-spend fields are not present in this source, so the model does not fabricate them.


## 📌 1. Nguồn gốc & Thu thập dữ liệu (Data Collection)
- **File dữ liệu thô:** `data/sales_data.csv`
- **Phương thức thu thập:** Được nạp tự động thông qua script `scripts/data_collection.py`.
- **Quy mô dữ liệu:** 500 bản ghi giao dịch (rows) và 11 thuộc tính gốc (columns).

---

## 📋 2. Giải thích chi tiết các thuộc tính (Data Dictionary)

| Tên cột | Kiểu dữ liệu | Mô tả ý nghĩa | Ví dụ mẫu |
| :--- | :--- | :--- | :--- |
| `order_id` | Integer | Mã định danh đơn hàng | `1`, `2` |
| `customer_id` | Integer | Mã định danh khách hàng | `150`, `176` |
| `product_id` | Integer | Mã định danh sản phẩm | `9`, `36` |
| `product_name` | String | Tên sản phẩm | `Tablet`, `Smartphone` |
| `category` | String | Danh mục sản phẩm (`Electronics`, `Accessories`, `Clothing`) | `Electronics` |
| `unit_price` | Integer | Đơn giá sản phẩm ($) | `611`, `751` |
| `quantity` | Integer | Số lượng mua trong đơn hàng | `1`, `4` |
| `order_date` | String/Date | Ngày thực hiện giao dịch (MM/DD/YYYY) | `11/14/2022` |
| `customer_age` | Integer | Độ tuổi của khách hàng | `62`, `23` |
| `customer_gender` | String | Giới tính khách hàng (`Male`, `Female`) | `Female` |
| `country` | String | Quốc gia khách hàng sinh sống | `Germany`, `Canada` |

---

## 🛠️ 3. Cách chạy script thu thập dữ liệu
Để tải/chép dữ liệu vào thư mục `data/`, mở terminal và chạy lệnh:

```bash
python scripts/data_collection.py
