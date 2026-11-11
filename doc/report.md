# Dự đoán doanh số sản phẩm — Báo cáo kết quả (Product Sales Prediction — Results Report)

## 1. Mục tiêu (Objective)

Xây dựng mô hình dự đoán doanh thu của từng giao dịch (`sales`) dựa trên các đặc trưng liên quan đến sản phẩm, khách hàng, giá bán, dữ liệu giao dịch trong quá khứ và thời gian.

## 2. Tập dữ liệu và biến mục tiêu (Dataset and Target)

- **Số lượng bản ghi hợp lệ:** 500 (từ 500 dòng dữ liệu ban đầu).
- **Phương pháp chia dữ liệu:** Chia theo thứ tự thời gian, gồm 409 bản ghi cho tập huấn luyện (Train) và 91 bản ghi cho tập kiểm tra (Test).
- **Ngày kết thúc tập huấn luyện:** 12/10/2022.
- **Ngày bắt đầu tập kiểm tra:** 13/10/2022.
- **Cách xác định biến mục tiêu:** `sales = unit_price * quantity`.
- **Lưu ý:** Thuộc tính `quantity` không được sử dụng làm đặc trưng đầu vào của mô hình nhằm tránh hiện tượng rò rỉ dữ liệu mục tiêu (Target Leakage), vì thuộc tính này được sử dụng trực tiếp để tính biến mục tiêu `sales`.

## 3. Phương pháp thực hiện (Method)

Nhóm thực hiện các bước xử lý và xây dựng mô hình như sau:

- **Xử lý dữ liệu thiếu:** Các giá trị số và giá trị phân loại bị thiếu được bổ sung ngay bên trong quy trình xử lý của mô hình (Model Pipeline).
- **Mã hóa dữ liệu:** Sử dụng phương pháp One-Hot Encoding để chuyển đổi các đặc trưng phân loại thành dạng số.
- **Làm sạch dữ liệu:** Loại bỏ các dòng dữ liệu trùng lặp và những bản ghi có giá bán, số lượng hoặc ngày giao dịch không hợp lệ.
- **Chia tập dữ liệu:** Sử dụng phương pháp chia theo thứ tự thời gian. Tập kiểm tra chứa các giao dịch có ngày phát sinh muộn hơn tập huấn luyện.
- **Xây dựng đặc trưng lịch sử:** Các đặc trưng như mức độ phổ biến của sản phẩm, tần suất mua hàng, giá trị vòng đời khách hàng và doanh thu trung bình trong quá khứ chỉ được tính dựa trên những giao dịch thuộc tập huấn luyện trước đó.
- **Xây dựng đặc trưng thời gian:** Trích xuất các thông tin gồm tháng, quý, thứ trong tuần, ngày cuối tuần và các đặc trưng mùa vụ có tính chu kỳ.

## 4. So sánh các mô hình (Model Comparison)

Nhóm tiến hành đánh giá và so sánh 5 mô hình dựa trên các chỉ số MAE, MSE, RMSE, R² và CV MAE.

| Mô hình | MAE | MSE | RMSE | R² | CV MAE |
|---|---:|---:|---:|---:|---:|
| RandomForest | 648.820 | 649599.435 | 805.977 | 0.286 | 611.947 |
| GradientBoosting | 653.421 | 625917.931 | 791.150 | 0.312 | 619.449 |
| DecisionTree | 743.484 | 960512.335 | 980.057 | -0.056 | 688.720 |
| LinearRegression | 671.922 | 698524.060 | 835.778 | 0.232 | 788.720 |
| DummyMean | 811.340 | 945125.931 | 972.176 | -0.039 | 854.401 |

**Nhận xét:**

- RandomForest đạt giá trị MAE thấp nhất trên tập kiểm tra và CV MAE thấp nhất trong quá trình kiểm định chéo theo chuỗi thời gian.
- GradientBoosting có chỉ số R² cao nhất (0.312) và RMSE thấp nhất (791.150).
- DecisionTree và DummyMean có R² âm, cho thấy khả năng dự đoán kém hơn so với mô hình dự đoán dựa trên giá trị trung bình trong trường hợp này.
- LinearRegression cho kết quả tốt hơn mô hình cơ sở DummyMean nhưng chưa đạt hiệu quả như hai mô hình RandomForest và GradientBoosting.

## 5. Mô hình được lựa chọn (Selected Model)

Nhóm lựa chọn **RandomForest** làm mô hình cuối cùng dựa trên tiêu chí có giá trị MAE thấp nhất trong quá trình kiểm định chéo theo chuỗi thời gian (Time-Series Cross-Validation).

Kết quả đánh giá trên tập kiểm tra độc lập (Chronological Test Set):

| Chỉ số | Kết quả |
|---|---:|
| MAE | 648.820 |
| MSE | 649599.435 |
| RMSE | 805.977 |
| R² | 0.286 |

Mặc dù GradientBoosting có R² cao hơn, RandomForest vẫn được lựa chọn vì đạt CV MAE thấp nhất, phù hợp với tiêu chí lựa chọn mô hình đã đặt ra.

## 6. Dự đoán thử nghiệm (Local Prediction)

Quy trình xử lý dữ liệu kết hợp với mô hình RandomForest đã được lưu tại:

`models/best_sales_model.joblib`

File này bao gồm cả quy trình tiền xử lý và mô hình đã huấn luyện, đồng thời sử dụng dữ liệu lịch sử phục vụ việc tạo các đặc trưng cần thiết.

Để chạy chương trình dự đoán thử nghiệm, thực hiện lệnh sau tại thư mục gốc của dự án:

```bash
python demo.py
```

Chương trình sẽ tải mô hình đã lưu và thực hiện dự đoán doanh thu cho một giao dịch mẫu.

## 7. Hạn chế và hướng phát triển (Limitations and Next Steps)

### Hạn chế của dữ liệu

Tập dữ liệu hiện tại chỉ cung cấp các thông tin về giá bán, số lượng sản phẩm, mã định danh sản phẩm, mã khách hàng, thông tin nhân khẩu học, quốc gia và ngày giao dịch trong khoảng thời gian từ 01/01/2022 đến 31/12/2022.

Dữ liệu không bao gồm:
- Điểm đánh giá và nhận xét sản phẩm (Reviews, Ratings).
- Chi phí quảng cáo (Advertising Spend).
- Thông tin các chiến dịch quảng cáo (Campaign Records).
- Nhãn xác định ngày lễ hoặc sự kiện cụ thể (Holiday Indicators).

Do đó, nhóm sử dụng số lượng giao dịch trong quá khứ làm chỉ số đại diện cho mức độ phổ biến của sản phẩm. Giá trị vòng đời khách hàng (Customer Lifetime Value) được ước tính thông qua tổng doanh thu tích lũy từ các giao dịch trước đó.

### Hạn chế trong phân tích thời gian

Phần phân tích dữ liệu khám phá (EDA) có xem xét doanh thu tổng hợp theo 12 tháng. Tuy nhiên, tập dữ liệu chỉ có một chu kỳ năm nên chưa đủ cơ sở để xác nhận một quy luật mùa vụ lặp lại hằng năm.

Vì vậy, nhóm không xây dựng các mô hình dự báo chuỗi thời gian như ARIMA hoặc Prophet do quy mô dữ liệu còn nhỏ và thời gian quan sát chưa đủ dài.

### Hạn chế về hệ thống gợi ý

Nhóm chưa triển khai hệ thống gợi ý sản phẩm (Recommendation System) vì tập dữ liệu không có thông tin phản hồi về mức độ yêu thích hoặc đánh giá của khách hàng đối với sản phẩm.

### Hướng phát triển

Trong tương lai, dự án có thể được cải thiện bằng cách:

- Thu thập thêm dữ liệu giao dịch trong nhiều năm để phân tích xu hướng và mùa vụ chính xác hơn.
- Bổ sung dữ liệu đánh giá và nhận xét sản phẩm.
- Tích hợp thông tin quảng cáo và các chiến dịch khuyến mãi.
- Thử nghiệm thêm các thuật toán Machine Learning và tối ưu tham số để cải thiện độ chính xác dự đoán.

## 8. Khả năng tái lập kết quả (Reproducibility)

Để chạy lại toàn bộ quy trình thực hiện dự án, sử dụng lệnh sau tại thư mục gốc:

```bash
python run_pipeline.py
```

Chương trình sẽ thực hiện các bước xử lý dữ liệu, huấn luyện và đánh giá mô hình.

Dự án sử dụng phương pháp chia dữ liệu theo thứ tự thời gian (Chronological Holdout) và kỹ thuật kiểm định chéo TimeSeriesSplit để lựa chọn tham số mô hình (Hyperparameter Selection).

**Lưu ý:** Tập kiểm tra (Test Set) được giữ riêng và chỉ sử dụng để đánh giá, so sánh kết quả cuối cùng, không sử dụng trong quá trình lựa chọn tham số mô hình.