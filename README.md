# Dự đoán doanh số sản phẩm (Product Sales Prediction)

Dự án xây dựng một quy trình Machine Learning hoàn chỉnh, có khả năng tái lập kết quả, nhằm dự đoán **doanh thu của từng giao dịch (Transaction Sales Revenue)** dựa trên các thông tin về sản phẩm, khách hàng, lịch sử giao dịch và thời gian.

## Biến mục tiêu và kiểm soát rò rỉ dữ liệu (Target and Leakage Control)

Biến mục tiêu được xác định theo công thức:

```python
sales = unit_price * quantity
```

Thuộc tính `quantity` được chủ động loại bỏ khỏi các đặc trưng đầu vào của mô hình. Nếu không, mô hình sẽ nhận được một thành phần trực tiếp cấu thành biến mục tiêu, khiến kết quả đánh giá trên tập kiểm tra không còn phản ánh đúng khả năng dự đoán.

Thay vào đó, mô hình ước tính doanh thu giao dịch dựa trên giá bán, thông tin sản phẩm, khách hàng, hành vi mua hàng trong quá khứ và các đặc trưng thời gian.

Tập dữ liệu được cung cấp gồm **500 giao dịch**, trong khoảng thời gian từ 01/01/2022 đến 31/12/2022. Dữ liệu bao gồm các thông tin về mã sản phẩm, mã khách hàng, tên và danh mục sản phẩm, đơn giá, số lượng, ngày giao dịch, độ tuổi, giới tính khách hàng và quốc gia.

Tập dữ liệu **không bao gồm** các thông tin như nhận xét sản phẩm, điểm đánh giá, chi phí quảng cáo, thông tin chiến dịch quảng cáo hoặc nhãn ngày lễ cụ thể.

Vì vậy, nhóm sử dụng:
- Số lượng giao dịch trong quá khứ để ước tính mức độ phổ biến của sản phẩm.
- Tổng doanh thu tích lũy từ các giao dịch trước đó để ước tính giá trị vòng đời khách hàng (Customer Lifetime Value).

Đây là những giá trị được ước tính dựa trên dữ liệu thực tế hiện có, không phải các thuộc tính được tự tạo hoặc giả định là có sẵn trong tập dữ liệu gốc.

## Cấu trúc dự án (Project Structure)

```text
ML_Lab2/
├── demo.py                         # Ví dụ dự đoán trên máy tính cá nhân
├── run_pipeline.py                 # Chạy toàn bộ quy trình dự án
├── data/
│   ├── raw/sales_data.csv
│   └── processed/                  # Dữ liệu đã làm sạch, chia tập và xây dựng đặc trưng
├── doc/
│   ├── description.md              # Mô tả và đề xuất dự án
│   └── report.md                   # Báo cáo được tạo từ kết quả chạy thực tế
├── figures/                        # Biểu đồ EDA và đánh giá mô hình
├── models/                         # Mô hình đã huấn luyện, chỉ số và kết quả dự đoán
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── training.py
│   ├── eval.py
│   ├── inference.py
│   ├── report.py
│   └── algorithms/                 # Các lớp mô hình được giữ lại từ dự án ban đầu
└── tests/test_pipeline.py
```

## Cài đặt môi trường (Setup)

Khuyến nghị sử dụng **Python 3.13** để tương thích với các tệp mô hình đã được lưu và cung cấp kèm theo dự án.

Các phiên bản thư viện chính xác được sử dụng trong quá trình huấn luyện mô hình đã được cố định trong file `requirements.txt`.

Tạo môi trường ảo và cài đặt các thư viện cần thiết bằng các lệnh sau:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Chạy toàn bộ dự án (Run the Complete Project)

Tại thư mục gốc của dự án, thực hiện lệnh:

```bash
python run_pipeline.py
```

### Các bước thực hiện trong Pipeline

1. **Kiểm tra dữ liệu:** Kiểm tra tính hợp lệ và sao chép tập dữ liệu gốc khi cần thiết.

2. **Phân tích dữ liệu khám phá (EDA):** Khám phá dữ liệu, lưu các biểu đồ và thống kê mô tả.

3. **Tiền xử lý dữ liệu:** Loại bỏ các dòng trùng lặp và bản ghi không hợp lệ; tính toán biến `sales`; kiểm tra và giới hạn các giá trị ngoại lệ của đặc trưng đầu vào bằng ngưỡng IQR được xác định chỉ từ tập huấn luyện.

4. **Chia tập dữ liệu:** Chia dữ liệu theo thứ tự thời gian, đảm bảo tập kiểm tra chứa các giao dịch phát sinh sau tập huấn luyện.

5. **Xây dựng đặc trưng (Feature Engineering):** Tạo các đặc trưng về mức độ phổ biến của sản phẩm, tần suất mua hàng của khách hàng, giá trị vòng đời khách hàng trong quá khứ, doanh thu trước đó của sản phẩm/khách hàng và các đặc trưng mùa vụ, thời gian. Quá trình này được thực hiện nhằm hạn chế rò rỉ dữ liệu.

6. **Huấn luyện mô hình:** Huấn luyện các mô hình Linear Regression, Decision Tree, Random Forest và Gradient Boosting. Sử dụng `GridSearchCV` kết hợp `TimeSeriesSplit` để tối ưu tham số cho các mô hình dựa trên cây quyết định. Kết quả được so sánh với mô hình cơ sở DummyRegressor.

7. **Đánh giá mô hình:** Đánh giá hiệu suất trên tập kiểm tra độc lập bằng các chỉ số MAE, MSE, RMSE và R².

8. **Lưu kết quả:** Lưu mô hình được lựa chọn, kết quả dự đoán, biểu đồ so sánh, mức độ quan trọng của các đặc trưng (Feature Importance) và báo cáo tại `doc/report.md`.

### Các tệp kết quả được tạo

Sau khi chạy Pipeline, các tệp kết quả chính bao gồm:

| Tệp | Nội dung |
|---|---|
| `models/metrics.csv` | Các chỉ số đánh giá mô hình |
| `models/tuning_results.csv` | Kết quả tối ưu tham số |
| `models/predictions.csv` | Kết quả dự đoán |
| `models/best_sales_model.joblib` | Mô hình được lựa chọn và Pipeline đã huấn luyện |
| `figures/model_comparison_mae.png` | Biểu đồ so sánh MAE giữa các mô hình |
| `figures/actual_vs_predicted.png` | Biểu đồ so sánh giá trị thực tế và giá trị dự đoán |

## Thực hiện dự đoán (Run a Prediction)

Sau khi hoàn thành quá trình huấn luyện, có thể sử dụng mô hình đã lưu để dự đoán trực tiếp trên máy tính cá nhân.

Tại thư mục gốc của dự án, chạy lệnh:

```bash
python demo.py
```

Chương trình sẽ:
- Tải mô hình đã lưu tại `models/best_sales_model.joblib`.
- Xây dựng các đặc trưng cần thiết từ giao dịch mẫu và dữ liệu lịch sử trong tập huấn luyện.
- Thực hiện dự đoán và hiển thị doanh thu của giao dịch dưới dạng giá trị số.

Để thử nghiệm với một giao dịch khác, người dùng có thể chỉnh sửa từ điển `sample` trong file `demo.py`, sau đó chạy lại câu lệnh trên.

## Chạy kiểm thử (Run Tests)

Để kiểm tra hoạt động của các thành phần trong dự án, sử dụng lệnh:

```bash
pytest -q
```

## Những hạn chế cần lưu ý (Important Limitations)

- Tập dữ liệu chỉ bao phủ khoảng 9 tháng của năm 2022, do đó chưa đủ cơ sở để kiểm chứng đáng tin cậy các dự báo mùa vụ hoặc ngày lễ theo chu kỳ hằng năm.
- Dữ liệu không có các thuộc tính về chiến dịch tiếp thị, chi phí quảng cáo, điểm đánh giá hoặc nhận xét sản phẩm. Vì vậy, các đặc trưng liên quan không thể được tính toán trực tiếp.
- Dự án hiện tại tập trung dự đoán doanh thu của từng giao dịch, không phải dự báo tổng doanh thu dài hạn cho từng sản phẩm. Các mô hình ARIMA/Prophet và hệ thống gợi ý sản phẩm (Recommendation Systems) được đề cập như những nội dung mở rộng tùy chọn trong bài tập, nhưng dữ liệu và biến mục tiêu hiện tại chưa hỗ trợ việc triển khai có ý nghĩa.
- Dự án được xây dựng phục vụ thực hành Machine Learning trên máy tính cá nhân, tập trung vào các bước chuẩn bị dữ liệu, xây dựng đặc trưng, huấn luyện mô hình, đánh giá và dự đoán thông qua các chương trình Python.

## Ghi chú về nguồn dữ liệu (Data Source Note)

Thông tin về nguồn gốc và các thuộc tính của tập dữ liệu được trình bày chi tiết trong file [`data/README.md`](data/README.md).

Nhóm không bổ sung thêm bản ghi dữ liệu hoặc tự tạo các thuộc tính về tiếp thị, quảng cáo không có trong tập dữ liệu gốc.