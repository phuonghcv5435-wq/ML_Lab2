# FINAL AUDIT — Practice 2: Product Sales Prediction

**Ngày rà soát tài liệu và repository:** 14/11/2026  
**Phạm vi:** Đối chiếu README, mã nguồn Python, dữ liệu và các artifact kết quả hiện có trên repository GitHub. Đây là rà soát tĩnh; notebook và toàn bộ pipeline chưa được thực thi lại trong môi trường máy tính của người dùng trong lần rà soát này.

## 1. Overall Assessment

Dự án đã có cấu trúc Machine Learning tương đối đầy đủ: định nghĩa Target, làm sạch dữ liệu, chia tập theo thời gian, tạo historical features, huấn luyện và tuning nhiều mô hình hồi quy, đánh giá trên hold-out Test, lưu model và inference cục bộ.

Kết quả hiện được lưu trong repository cho thấy **Random Forest** là mô hình được chọn theo CV MAE thấp nhất. Tuy nhiên, R² Test khoảng 0.286 cho thấy khả năng giải thích biến thiên doanh thu còn khiêm tốn. Vì vậy, dự án phù hợp với mục tiêu học tập và minh họa workflow, chưa nên mô tả là một hệ thống dự đoán doanh thu có độ chính xác cao hoặc sẵn sàng dùng trong vận hành.

Không đưa ra phần trăm hoàn thành hoặc điểm số giảng viên vì không có rubric chấm điểm chính thức trong phạm vi rà soát này.

## 2. Requirement Coverage

| Hạng mục | Trạng thái | Nhận xét |
|---|---|---|
| Problem definition | PASS | Target là doanh thu giao dịch, được định nghĩa rõ bằng `unit_price * quantity`. |
| Dataset structure | PASS | Có 500 bản ghi và 11 cột nguồn theo `data/raw/sales_data.csv`. |
| Data cleaning | PASS | Có parse ngày, ép kiểu số, loại duplicate và bỏ dòng thiếu/không hợp lệ ở các trường trọng yếu. |
| Missing-value handling | PASS | Các trường trọng yếu được kiểm tra khi cleaning; các cột đầu vào còn thiếu được xử lý trong preprocessing pipeline. |
| Outlier handling | PASS | Ngưỡng IQR được tính từ Train cho `unit_price` và `customer_age`, rồi áp dụng cho Test; Target không bị cắt. |
| Chronological split | PASS | 409 Train và 91 Test; Train kết thúc 12/10/2022, Test bắt đầu 13/10/2022. |
| Target leakage control | PASS | `quantity` không có trong Feature DataFrame cuối; `training.py` có kiểm tra `quantity` và `sales` không được xuất hiện trong X. |
| Product history features | PASS | Có số giao dịch trước đó và doanh thu trung bình trước đó của sản phẩm. |
| Customer history features | PASS | Có tần suất mua, tổng doanh thu tích lũy trước đó và doanh thu trung bình trước đó. |
| Temporal features | PASS | Có tháng, quý, thứ trong tuần, cuối tuần và encoding chu kỳ. |
| Historical leakage control | PASS | Thứ tự tạo feature là lấy lịch sử trước, sau đó mới cập nhật bằng giao dịch Train hiện tại; Test dùng lịch sử Train ở chế độ chỉ đọc. |
| Regression algorithms | PASS | Linear Regression, Decision Tree, Random Forest và Gradient Boosting. |
| Baseline | PASS | Có DummyRegressor theo giá trị trung bình. |
| Hyperparameter tuning | PASS | GridSearchCV kết hợp TimeSeriesSplit cho các mô hình cây. |
| Model selection | PASS | Chọn theo CV MAE thấp nhất trên Train; Test giữ cho đánh giá cuối cùng. |
| Evaluation | PASS | Có MAE, MSE, RMSE, R² và CV_MAE. |
| Visualizations | PASS | Có các biểu đồ EDA, so sánh mô hình, Actual vs Predicted và Feature Importance. |
| Model persistence | PASS | Lưu các mô hình, best model bundle và lịch sử feature. |
| Local inference | PASS — code reviewed | `demo.py` gọi `predict_sales()` và sử dụng model bundle đã lưu; cần chạy lại trên máy cá nhân để xác nhận môi trường thực tế. |
| Automated tests | IMPLEMENTED — run required | Có `tests/test_pipeline.py` với các kiểm tra Target, chronological split, historical features và inference history. Cần chạy `pytest -q` để xác nhận kết quả tại máy hiện tại. |
| Notebook | ADDED IN THIS DOCUMENT PACK | Notebook được tạo để điều phối cùng workflow; chưa có output thực thi được lưu sẵn nhằm tránh đưa kết quả giả vào báo cáo. |
| API/web deployment | OUT OF SCOPE | Dự án giữ ở dạng Python local, phù hợp yêu cầu không sử dụng API. |

## 3. Dataset Audit

Kết quả trong `data/processed/preprocessing_summary.json`:

| Chỉ tiêu | Giá trị |
|---|---:|
| Original rows | 500 |
| Clean rows | 500 |
| Removed rows | 0 |
| Train rows | 409 |
| Test rows | 91 |
| Date range | 2022-01-01 đến 2022-12-31 |
| Train end date | 2022-10-12 |
| Test start date | 2022-10-13 |

Trong lần chạy được lưu lại, không có bản ghi nào bị loại. Điều này không có nghĩa các bước cleaning là không cần thiết; chúng vẫn giúp bảo vệ pipeline khi dữ liệu đầu vào thay đổi.

### Dataset limitations

- Dataset có 500 giao dịch, quy mô nhỏ để khái quát hóa sang hoạt động kinh doanh rộng hơn.
- Dữ liệu bao phủ một năm, không đủ để xác nhận một mô hình mùa vụ lặp lại giữa nhiều năm.
- Không có quảng cáo, chi phí marketing, chiến dịch, review/rating hoặc nhãn ngày lễ chính thức.
- `is_holiday_season` chỉ đánh dấu tháng 11 và 12 như proxy mùa cao điểm.
- `customer_lifetime_value_prior` là tổng doanh thu quan sát được trong lịch sử trước đó, không phải ước lượng CLV trọn đời.

## 4. Model Results Audit

Dữ liệu từ `models/metrics.csv` và `models/training_summary.json`:

| Model | CV MAE | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| RandomForest | **611.9466** | **648.8198** | 649,599.4352 | 805.9773 | 0.2857 |
| GradientBoosting | 619.4489 | 653.4212 | **625,917.9312** | **791.1498** | **0.3117** |
| DecisionTree | 688.7202 | 743.4835 | 960,512.3346 | 980.0573 | -0.0562 |
| LinearRegression | 788.7197 | 671.9223 | 698,524.0605 | 835.7775 | 0.2319 |
| DummyMean | 854.4013 | 811.3395 | 945,125.9309 | 972.1759 | -0.0393 |

### Interpretation

- Random Forest được chọn do có CV MAE thấp nhất (611.9466).
- Random Forest có Test MAE thấp nhất trong bảng.
- Gradient Boosting có Test MSE/RMSE thấp hơn và R² cao hơn Random Forest.
- Các mô hình học được đều tốt hơn Dummy Mean ở CV MAE và Test MAE trong kết quả được lưu.
- R² của Random Forest là 0.2857, do đó hiệu quả giải thích biến thiên doanh thu còn hạn chế.

Không nên chỉ trình bày rằng Random Forest “tốt nhất ở mọi chỉ số”. Cách diễn giải chính xác là: Random Forest được chọn theo tiêu chí CV MAE đã định trước; Gradient Boosting đạt một số chỉ số Test tốt hơn.

## 5. Technical Review

### Data leakage

`quantity` tham gia công thức tạo Target nhưng không được đưa vào Feature DataFrame cuối. Các historical features được tạo theo thứ tự thời gian; Test sử dụng history từ Train và không cập nhật history bằng target Test. Đây là các điểm thiết kế quan trọng để hạn chế rò rỉ dữ liệu.

### Model validation

`TimeSeriesSplit(n_splits=3)` được dùng trong tập Train. Tập Test là giai đoạn thời gian sau cùng, không tham gia lựa chọn tham số hoặc mô hình.

### Model artifact

`models/best_sales_model.joblib` chứa model được chọn, tên model, Target, danh sách feature columns, lịch sử sản phẩm/khách hàng và thông tin về việc loại `quantity` khỏi Feature. `src/inference.py` dùng bundle này để tạo feature cho giao dịch mới và căn chỉnh thứ tự cột.

### Reproducibility

`requirements.txt` cố định các phiên bản thư viện. Khi thay đổi dữ liệu, thư viện hoặc mã nguồn, cần chạy lại toàn bộ pipeline để tạo một bộ kết quả đồng nhất.

## 6. Issues and Recommendations

### Issue 1 — Performance is moderate

**Observation:** R² Test của Random Forest khoảng 0.286.

**Recommendation:** Thu thập thêm giao dịch, thử thêm giai đoạn kiểm tra theo thời gian, phân tích residual và kiểm tra sai số theo danh mục/quốc gia. Không nên tự ý tăng độ phức tạp mô hình khi dữ liệu vẫn quá nhỏ.

### Issue 2 — Business interpretation of Target

**Observation:** `quantity` bị loại khỏi X nên mô hình không nhận đầu vào số lượng dự kiến.

**Recommendation:** Giữ cách thiết kế hiện tại nếu mục tiêu học tập là tránh mô hình học lại công thức tạo Target. Nếu bài toán kinh doanh thực tế cần dự đoán doanh thu theo số lượng đã biết, cần định nghĩa lại mục tiêu thực nghiệm và cách đánh giá cho phù hợp thay vì âm thầm thêm `quantity` vào X.

### Issue 3 — Limited external business signals

**Observation:** Không có marketing, campaign, review/rating hoặc holiday labels.

**Recommendation:** Chỉ bổ sung các trường này khi có dữ liệu nguồn hợp lệ. Không tự điền dữ liệu giả và mô tả như quan sát thực tế.

### Issue 4 — Notebook execution status

**Observation:** Notebook trong gói tài liệu này chưa có output được thực thi sẵn.

**Recommendation:** Mở notebook trong môi trường đã cài `requirements.txt`, chạy từ trên xuống, xác nhận các cell thành công rồi lưu notebook nếu cần nộp kèm output.

## 7. Final Verdict

Dự án có đầy đủ các thành phần cốt lõi của một bài thực hành regression: xử lý dữ liệu, chronological split, feature engineering có kiểm soát lịch sử, nhiều thuật toán, tuning, cross-validation, đánh giá hold-out, lưu model và inference cục bộ.

Điểm cần nhấn mạnh khi báo cáo là kết quả còn ở mức thực nghiệm: Random Forest được chọn nhờ CV MAE tốt nhất nhưng R² Test khoảng 0.286; dữ liệu có quy mô nhỏ và chỉ bao phủ một năm. Mọi nhận xét về khả năng ứng dụng cần đặt trong các giới hạn này.

## 8. Reproducibility Checklist

- [ ] Tạo môi trường Python tương thích và cài `requirements.txt`.
- [ ] Chạy `python run_pipeline.py` từ thư mục gốc.
- [ ] Kiểm tra `data/processed/preprocessing_summary.json`.
- [ ] Kiểm tra `models/metrics.csv` và `models/training_summary.json`.
- [ ] Kiểm tra model bundle `models/best_sales_model.joblib`.
- [ ] Chạy `python demo.py`.
- [ ] Chạy `pytest -q`.
- [ ] Cập nhật tài liệu nếu kết quả mới khác kết quả đang ghi trong README.
