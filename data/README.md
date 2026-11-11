# Dataset (Tập dữ liệu)

Dự án sử dụng tập dữ liệu giao dịch bán hàng `data/raw/sales_data.csv`, gồm **500 dòng dữ liệu**, được cung cấp trong repository ban đầu.

Theo ghi chú của repository gốc, dữ liệu được lấy từ một Gist công khai chứa dữ liệu bán hàng mô phỏng và được cho là bao gồm **150 giao dịch đầu tiên**. Tuy nhiên, file `sales_data.csv` thực tế có trong file ZIP được cung cấp lại chứa **500 dòng dữ liệu**.

Sự khác biệt này được giữ lại như một ghi chú về nguồn gốc dữ liệu. Trong phạm vi dự án, nhóm sử dụng toàn bộ **500 dòng dữ liệu thực tế** có trong file ZIP đã nhận.

**Nguồn dữ liệu được ghi nhận:**  
[https://gist.github.com/summerofgeorge/33dbb09adbdea670c31ee2c4fc9f2617](https://gist.github.com/summerofgeorge/33dbb09adbdea670c31ee2c4fc9f2617)

**Khoảng thời gian dữ liệu:** Từ **01/01/2022 đến 31/12/2022**.

## Các thuộc tính có trong tập dữ liệu

| Tên cột | Ý nghĩa |
|---|---|
| `order_id` | Mã định danh giao dịch |
| `customer_id` | Mã định danh khách hàng |
| `product_id` | Mã định danh sản phẩm |
| `product_name` | Tên sản phẩm |
| `category` | Danh mục sản phẩm |
| `unit_price` | Đơn giá của sản phẩm |
| `quantity` | Số lượng sản phẩm được mua; chỉ sử dụng để tính biến mục tiêu, không sử dụng làm đặc trưng đầu vào của mô hình |
| `order_date` | Ngày thực hiện giao dịch |
| `customer_age` | Độ tuổi của khách hàng |
| `customer_gender` | Giới tính khách hàng theo thông tin được cung cấp trong dữ liệu |
| `country` | Quốc gia của khách hàng |

## Biến mục tiêu (Target Variable)

Biến mục tiêu được xây dựng bằng cách tính tổng doanh thu của mỗi giao dịch theo công thức:

`sales = unit_price * quantity`

Trong đó:
- `unit_price`: Đơn giá của sản phẩm.
- `quantity`: Số lượng sản phẩm được mua.
- `sales`: Tổng doanh thu của giao dịch.

## Lưu ý về dữ liệu

Tập dữ liệu được cung cấp không có các thuộc tính như:
- Điểm đánh giá sản phẩm (Ratings).
- Số lượng đánh giá (Review Counts).
- Chi phí quảng cáo (Advertising Spend).
- Mã chiến dịch quảng cáo (Campaign Identifiers).
- Nhãn sự kiện hoặc ngày lễ cụ thể (Holiday/Event Labels).

Do đó, trong phạm vi dự án, nhóm sử dụng **số lượng giao dịch và doanh thu trong quá khứ** làm các chỉ số đại diện (Proxies) để đánh giá mức độ phổ biến của sản phẩm và giá trị khách hàng.

Cách tiếp cận này giúp nhóm khai thác những thông tin thực tế có sẵn trong dữ liệu mà không tự giả định hoặc bổ sung các thuộc tính không tồn tại.