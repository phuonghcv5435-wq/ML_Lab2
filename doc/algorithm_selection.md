# Algorithm Selection — Practice 2: Product Sales Prediction

## 1. Problem Type

Bài toán dự đoán doanh thu giao dịch là **Supervised Learning — Regression**.

Target là một giá trị số liên tục:

```python
sales = unit_price * quantity
```

Mục tiêu của mô hình là ước lượng `sales` cho giao dịch mới dựa trên thông tin sản phẩm, khách hàng, đơn giá, lịch sử giao dịch và thời gian.

`quantity` được dùng để tạo Target nhưng không được đưa vào tập Feature cuối cùng. Đây là lựa chọn thiết kế để hạn chế mô hình học lại trực tiếp công thức tạo Target. Hệ quả là mô hình không trả lời kịch bản “đã biết số lượng dự kiến và muốn tính tổng doanh thu cho số lượng đó”.

## 2. Feature Groups

Các Feature được tạo trong `src/feature_engineering.py` gồm:

| Nhóm | Feature tiêu biểu | Lý do sử dụng |
|---|---|---|
| Product | `product_id`, `product_name`, `category` | Mô tả sản phẩm và nhóm hàng |
| Customer | `customer_id`, `customer_age`, `customer_gender`, `country` | Mô tả khách hàng |
| Product history | `product_popularity_prior`, `product_avg_sales_prior` | Phản ánh lịch sử giao dịch trước đó của sản phẩm |
| Customer history | `customer_purchase_frequency`, `customer_lifetime_value_prior`, `customer_avg_sales_prior` | Phản ánh lịch sử mua hàng của khách hàng |
| Calendar | `month`, `quarter`, `day_of_week`, `is_weekend` | Mô tả thời điểm giao dịch |
| Seasonality | `is_holiday_season`, `month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos` | Mô tả mùa cao điểm và chu kỳ thời gian |

Các Feature lịch sử được tính theo thứ tự giao dịch. Với một giao dịch, thống kê được lấy trước, sau đó lịch sử mới được cập nhật bằng doanh thu của giao dịch đó. Tập Test dùng lịch sử Train ở chế độ chỉ đọc.

`is_holiday_season` được định nghĩa như proxy tháng 11–12. Dataset không có nhãn ngày lễ chính thức.

## 3. Candidate Algorithms

### 3.1. Dummy Regressor (Mean Baseline)

**Vai trò:** Mô hình tham chiếu đơn giản, luôn dự đoán giá trị trung bình của Target trong tập huấn luyện.

**Lý do sử dụng:** Cần một baseline để biết các mô hình phức tạp có thực sự cải thiện so với cách dự đoán đơn giản hay không.

### 3.2. Linear Regression

**Nguyên lý:** Mô hình hóa quan hệ giữa các Feature và Target dưới dạng kết hợp tuyến tính.

**Lý do lựa chọn:**
- Là baseline Machine Learning dễ hiểu.
- Dễ diễn giải và so sánh.
- Giúp kiểm tra xem một quan hệ gần tuyến tính có đủ để dự đoán doanh thu hay không.

**Hạn chế:** Khó mô tả các tương tác và quan hệ phi tuyến phức tạp nếu không thiết kế Feature phù hợp.

### 3.3. Decision Tree Regressor

**Nguyên lý:** Chia dữ liệu thành các nhánh theo điều kiện trên Feature và dự đoán giá trị trong từng vùng.

**Lý do lựa chọn:**
- Có khả năng học quan hệ phi tuyến.
- Không yêu cầu quan hệ đầu vào–đầu ra tuyến tính.
- Có thể nắm bắt tương tác giữa các biến.

**Hạn chế:** Cây đơn dễ học quá sát tập Train, đặc biệt khi dữ liệu ít.

### 3.4. Random Forest Regressor

**Nguyên lý:** Kết hợp nhiều cây quyết định được huấn luyện trên các mẫu và tập Feature ngẫu nhiên khác nhau; dự đoán cuối cùng là tổng hợp dự đoán từ các cây.

**Lý do lựa chọn:**
- Thường ổn định hơn một cây quyết định đơn.
- Học được quan hệ phi tuyến và tương tác giữa các Feature.
- Phù hợp làm ứng viên mạnh cho dữ liệu bảng.

**Hạn chế:** Khó diễn giải toàn bộ mô hình; mô hình lớn hơn cây đơn và vẫn cần dữ liệu đủ đại diện.

### 3.5. Gradient Boosting Regressor

**Nguyên lý:** Huấn luyện các cây theo tuần tự, mỗi cây mới tập trung cải thiện sai số còn lại của mô hình trước.

**Lý do lựa chọn:**
- Có thể mô hình hóa quan hệ phi tuyến.
- Có khả năng đạt hiệu quả tốt trên dữ liệu bảng.
- Cung cấp một hướng ensemble khác với Random Forest để so sánh.

**Hạn chế:** Nhạy với cấu hình tham số và có thể overfit nếu tuning không phù hợp.

## 4. Hyperparameter Tuning

Dự án sử dụng `GridSearchCV` với `TimeSeriesSplit(n_splits=3)` cho các mô hình cây. Các lưới tham số hiện có trong `src/training.py`:

### Decision Tree

| Parameter | Giá trị thử nghiệm |
|---|---|
| `max_depth` | `None`, `5`, `10` |
| `min_samples_leaf` | `1`, `3` |

### Random Forest

| Parameter | Giá trị thử nghiệm |
|---|---|
| `n_estimators` | `100` |
| `max_depth` | `None`, `10` |
| `min_samples_leaf` | `1`, `3` |

### Gradient Boosting

| Parameter | Giá trị thử nghiệm |
|---|---|
| `n_estimators` | `100` |
| `learning_rate` | `0.03`, `0.1` |
| `max_depth` | `2`, `3` |

Linear Regression không có lưới tham số trong phiên bản hiện tại; mô hình được đánh giá bằng cross-validation.

## 5. Preprocessing Strategy

Mỗi mô hình được ghép với một preprocessing pipeline:

- Cột số: `SimpleImputer(strategy="median")` và `StandardScaler`.
- Cột phân loại: `SimpleImputer(strategy="most_frequent")` và `OneHotEncoder(handle_unknown="ignore")`.

Pipeline được lưu chung với estimator để đảm bảo quy trình biến đổi dữ liệu được áp dụng nhất quán khi inference.

## 6. Validation Strategy

Dữ liệu được sắp xếp theo `order_date` và chia thành hai giai đoạn:

- **Train:** 409 giao dịch, giao dịch cuối cùng ngày 12/10/2022.
- **Test:** 91 giao dịch, giao dịch đầu tiên ngày 13/10/2022.

Trong tập Train, `TimeSeriesSplit(n_splits=3)` được dùng để đánh giá và tuning. Cách chia này giữ thứ tự thời gian: các fold đánh giá nằm sau phần dữ liệu dùng để huấn luyện fold tương ứng.

Tập Test được giữ lại để báo cáo hiệu quả cuối cùng, không tham gia chọn tham số hoặc quyết định mô hình tốt nhất.

## 7. Evaluation Metrics

| Metric | Ý nghĩa | Hướng tốt hơn |
|---|---|---|
| MAE | Sai số tuyệt đối trung bình | Càng thấp càng tốt |
| MSE | Sai số bình phương trung bình | Càng thấp càng tốt |
| RMSE | Căn bậc hai của MSE, cùng đơn vị với doanh thu | Càng thấp càng tốt |
| R² | Mức độ giải thích biến thiên của Target | Càng gần 1 càng tốt |
| CV_MAE | MAE trung bình qua các fold thời gian trên Train | Càng thấp càng tốt |

## 8. Results and Selection

Kết quả hiện được lưu trong `models/metrics.csv`:

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| **Random Forest** | **611.95** | **648.82** | 805.98 | 0.2857 |
| Gradient Boosting | 619.45 | 653.42 | **791.15** | **0.3117** |
| Decision Tree | 688.72 | 743.48 | 980.06 | -0.0562 |
| Linear Regression | 788.72 | 671.92 | 835.78 | 0.2319 |
| Dummy Mean | 854.40 | 811.34 | 972.18 | -0.0393 |

**Selected model: Random Forest Regressor.**

Random Forest được chọn vì có `CV_MAE` thấp nhất (611.95) trong quy trình validation trên tập Train. Cấu hình tốt nhất được ghi trong `models/tuning_results.csv` là `n_estimators=100`, `max_depth=10`, `min_samples_leaf=1`. Không chọn mô hình dựa trên kết quả Test vì Test được dành cho đánh giá cuối cùng.

Cần diễn giải kết quả đầy đủ: Gradient Boosting có Test RMSE thấp hơn và Test R² cao hơn Random Forest, trong khi Random Forest có Test MAE thấp hơn và CV MAE tốt hơn. Vì quy tắc lựa chọn đã định trước là CV MAE thấp nhất, Random Forest vẫn là mô hình được chọn trong lần chạy này.

R² Test của Random Forest khoảng 0.286, do đó mô hình chưa thể được xem là có độ chính xác cao hoặc sẵn sàng triển khai thực tế. Kết quả có giá trị trong phạm vi thực hành và so sánh thuật toán trên dataset hiện tại.

## 9. Why Not Other Approaches?

- **Classification:** Không phù hợp vì Target là giá trị doanh thu liên tục, không phải nhãn lớp.
- **ARIMA/Prophet:** Dự án hiện dự đoán doanh thu theo từng giao dịch; dữ liệu chỉ có một năm nên chưa đủ chu kỳ để đánh giá mùa vụ lặp lại đáng tin cậy.
- **Recommendation System:** Dataset không có rating, review hoặc phản hồi sở thích phù hợp để xây dựng hệ thống gợi ý có ý nghĩa.
- **Marketing-response model:** Dataset không có advertising spend hoặc campaign records, nên không thể đánh giá tác động của quảng cáo.

## 10. Future Experiments

1. Bổ sung dữ liệu nhiều năm để đánh giá xu hướng mùa vụ.
2. Mở rộng lưới tuning và thử nhiều mốc thời gian hold-out.
3. Phân tích sai số theo nhóm sản phẩm, danh mục, quốc gia và giai đoạn.
4. Thử nghiệm các mô hình boosting khác nếu có đủ dữ liệu và tài nguyên.
5. Đánh giá lại cách định nghĩa Target nếu bài toán kinh doanh yêu cầu dự đoán doanh thu theo một số lượng đặt trước.

## 11. Implementation References

- `src/data_preprocessing.py`
- `src/feature_engineering.py`
- `src/training.py`
- `models/metrics.csv`
- `models/tuning_results.csv`
- `models/training_summary.json`
