# Practice 2 — Product Sales Prediction

**Dự án Machine Learning dự đoán doanh thu của từng giao dịch sản phẩm**

Dự án xây dựng một quy trình Machine Learning hoàn chỉnh để ước lượng doanh thu (`sales`) của một giao dịch dựa trên thông tin sản phẩm, khách hàng, giá bán, lịch sử giao dịch và thời gian. Đây là bài toán **Supervised Learning — Regression**.

> Dự án chạy bằng Python trên máy tính cá nhân. Không sử dụng FastAPI, REST API hoặc dịch vụ triển khai web.

---

## 1. Project Objective

Mục tiêu của dự án là thực hiện đầy đủ quy trình xây dựng một mô hình hồi quy:

1. Đọc và kiểm tra dữ liệu giao dịch.
2. Phân tích khám phá dữ liệu (EDA).
3. Làm sạch dữ liệu và chia tập huấn luyện/kiểm tra theo thời gian.
4. Tạo đặc trưng từ thông tin sản phẩm, khách hàng, lịch sử mua hàng và thời gian.
5. Huấn luyện, tuning và so sánh nhiều mô hình hồi quy.
6. Đánh giá mô hình trên tập kiểm tra độc lập.
7. Lưu mô hình tốt nhất và sử dụng lại để dự đoán giao dịch mới.

### 1.1. Problem Definition

Biến mục tiêu là doanh thu của một giao dịch:

```python
sales = unit_price * quantity
```

Trong đó:

- `unit_price`: đơn giá của sản phẩm.
- `quantity`: số lượng sản phẩm trong giao dịch.
- `sales`: doanh thu của giao dịch.

`quantity` được sử dụng để tính Target nhưng được loại khỏi tập đặc trưng đầu vào cuối cùng. Đây là lựa chọn thiết kế nhằm tránh để mô hình chỉ học lại công thức tạo Target. Đổi lại, mô hình hiện tại **không phù hợp với kịch bản người dùng nhập trước một số lượng dự kiến rồi yêu cầu tính doanh thu cho số lượng đó**.

Bài toán của dự án là ước lượng doanh thu từ các thuộc tính còn lại và thông tin lịch sử có sẵn, không phải dự đoán số lượng sản phẩm bán ra.

## 2. Dataset

### 2.1. Dataset Overview

Dữ liệu nằm tại `data/raw/sales_data.csv` và gồm **500 giao dịch**, với 11 cột ban đầu. Khoảng thời gian trong dữ liệu là từ **01/01/2022 đến 31/12/2022**.

Theo kết quả tiền xử lý hiện được lưu trong repository:

| Thông tin | Kết quả |
|---|---:|
| Số dòng ban đầu | 500 |
| Số dòng sau làm sạch | 500 |
| Số dòng Train | 409 |
| Số dòng Test | 91 |
| Ngày cuối Train | 2022-10-12 |
| Ngày đầu Test | 2022-10-13 |

Tập Test chứa các giao dịch ở giai đoạn thời gian sau tập Train. Vì chia theo ngày giao dịch, số lượng bản ghi thực tế không nhất thiết đúng chính xác tỷ lệ 80/20.

### 2.2. Dataset Features

| Cột | Ý nghĩa | Vai trò |
|---|---|---|
| `order_id` | Mã giao dịch | Định danh, không đưa vào Feature |
| `customer_id` | Mã khách hàng | Tạo đặc trưng lịch sử và danh mục |
| `product_id` | Mã sản phẩm | Tạo đặc trưng lịch sử và danh mục |
| `product_name` | Tên sản phẩm | Feature phân loại |
| `category` | Danh mục sản phẩm | Feature phân loại |
| `unit_price` | Đơn giá | Feature số |
| `quantity` | Số lượng mua | Dùng tạo Target, không đưa vào X cuối cùng |
| `order_date` | Ngày giao dịch | Tạo đặc trưng thời gian |
| `customer_age` | Tuổi khách hàng | Feature số |
| `customer_gender` | Giới tính khách hàng | Feature phân loại |
| `country` | Quốc gia | Feature phân loại |

### 2.3. Data Limitations

Dataset hiện không có các trường như đánh giá sản phẩm, nội dung nhận xét, chi phí quảng cáo, mã chiến dịch marketing hoặc nhãn ngày lễ chính thức. Vì vậy, dự án không tự tạo dữ liệu cho những trường này.

Dữ liệu chỉ bao phủ một năm. Các đặc trưng mùa vụ có thể giúp mô hình nhận biết thời điểm trong năm, nhưng một chu kỳ năm duy nhất chưa đủ để xác nhận xu hướng lặp lại qua nhiều năm.

## 3. Machine Learning Workflow

```text
Raw Dataset
    │
    ▼
Data Collection
    │
    ▼
Exploratory Data Analysis (EDA)
    │
    ▼
Data Cleaning & Validation
    │
    ▼
Chronological Train/Test Split
    │
    ▼
Feature Engineering
    │
    ▼
Preprocessing Pipeline
(Imputation + Scaling + One-Hot Encoding)
    │
    ▼
Model Training & Hyperparameter Tuning
    │
    ▼
TimeSeriesSplit Cross-Validation
    │
    ▼
Final Evaluation on Hold-out Test Set
    │
    ▼
Save Best Model
    │
    ▼
Local Prediction Demo
```

### 3.1. Data Collection

Module `src/data_collection.py` đọc dữ liệu từ `data/raw/sales_data.csv` và chuẩn bị thư mục dữ liệu đầu vào cho các bước tiếp theo.

### 3.2. Exploratory Data Analysis (EDA)

Module `src/eda.py` kiểm tra tổng quan dữ liệu và tạo các biểu đồ:

- Phân phối doanh thu giao dịch.
- Tổng doanh thu theo danh mục sản phẩm.
- Tổng doanh thu theo quốc gia.
- Xu hướng doanh thu theo tháng.
- Mối quan hệ giữa đơn giá và doanh thu giao dịch.

Các biểu đồ được lưu trong `figures/`; thống kê tổng quan được lưu trong `figures/eda_summary.json`.

### 3.3. Data Preprocessing

Module `src/data_preprocessing.py` thực hiện:

1. Chuyển `order_date` sang kiểu ngày tháng.
2. Chuyển các cột số quan trọng sang dạng số.
3. Loại bỏ bản ghi trùng lặp.
4. Loại các bản ghi thiếu ngày, đơn giá hoặc số lượng hợp lệ.
5. Kiểm tra các giá trị đơn giá và số lượng không hợp lệ.
6. Tạo Target `sales = unit_price * quantity`.
7. Chia Train/Test theo thứ tự thời gian.
8. Tính ngưỡng IQR từ tập Train để giới hạn ngoại lệ của các biến đầu vào số; các ngưỡng này được áp dụng cho cả Train và Test. Target không bị cắt ngưỡng.

Trong lần chạy có kết quả được lưu trong repository, dữ liệu có 500 dòng và không có dòng nào bị loại ở bước làm sạch. Tập Train có 409 dòng, tập Test có 91 dòng.

### 3.4. Feature Engineering

Module `src/feature_engineering.py` tạo đặc trưng mà không dùng thông tin tương lai của giao dịch đang dự đoán.

| Feature | Ý nghĩa |
|---|---|
| `product_popularity_prior` | Số giao dịch trước đó của sản phẩm |
| `product_avg_sales_prior` | Doanh thu trung bình trước đó của sản phẩm |
| `customer_purchase_frequency` | Số giao dịch trước đó của khách hàng |
| `customer_lifetime_value_prior` | Tổng doanh thu tích lũy của khách hàng trước giao dịch hiện tại |
| `customer_avg_sales_prior` | Doanh thu giao dịch trung bình trước đó của khách hàng |
| `month` | Tháng giao dịch |
| `quarter` | Quý giao dịch |
| `day_of_week` | Thứ trong tuần |
| `is_weekend` | Cờ cuối tuần |
| `is_holiday_season` | Proxy mùa cao điểm, đánh dấu tháng 11 và 12 |
| `month_sin`, `month_cos` | Mã hóa chu kỳ tháng |
| `day_of_year_sin`, `day_of_year_cos` | Mã hóa chu kỳ ngày trong năm |

Các đặc trưng lịch sử được tính từ những giao dịch đứng trước. Khi xử lý Test, module sử dụng lịch sử từ Train ở chế độ chỉ đọc, không cập nhật lịch sử bằng nhãn của Test.

`customer_lifetime_value_prior` trong bài này là tổng doanh thu lịch sử quan sát được trước giao dịch hiện tại, không phải CLV dự báo trọn đời theo nghĩa tài chính/marketing.

### 3.5. Preprocessing Pipeline

Trong `src/training.py`, các bước xử lý đầu vào được đóng gói cùng mô hình bằng `sklearn.pipeline.Pipeline`:

- Biến số: điền giá trị thiếu bằng Median và chuẩn hóa bằng `StandardScaler`.
- Biến phân loại: điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất và mã hóa bằng `OneHotEncoder(handle_unknown="ignore")`.

Việc lưu cả bước tiền xử lý và estimator trong cùng Pipeline giúp dữ liệu mới được chuyển đổi nhất quán khi dự đoán.

### 3.6. Model Training & Hyperparameter Tuning

Dự án so sánh bốn mô hình hồi quy:

1. Linear Regression.
2. Decision Tree Regressor.
3. Random Forest Regressor.
4. Gradient Boosting Regressor.

Ngoài ra, `DummyRegressor(strategy="mean")` được dùng làm baseline để kiểm tra các mô hình học máy có cải thiện so với cách dự đoán bằng giá trị trung bình hay không.

Các mô hình cây được tuning bằng `GridSearchCV` kết hợp `TimeSeriesSplit(n_splits=3)`. Linear Regression và Dummy baseline cũng được đánh giá bằng cross-validation trên tập Train.

### 3.7. Model Evaluation & Selection

Các chỉ số được sử dụng:

- **MAE:** Sai số tuyệt đối trung bình; càng thấp càng tốt.
- **MSE:** Sai số bình phương trung bình; càng thấp càng tốt và phạt mạnh sai số lớn.
- **RMSE:** Căn bậc hai của MSE; cùng đơn vị với Target.
- **R²:** Mức độ giải thích biến thiên của Target; càng gần 1 càng tốt. R² âm có nghĩa mô hình kém hơn baseline dự đoán trung bình trên tập đánh giá.
- **CV_MAE:** MAE trung bình từ cross-validation trên tập Train.

Quy tắc chọn mô hình là **CV_MAE thấp nhất**. Tập Test được giữ lại để đánh giá cuối cùng, không dùng để chọn tham số hoặc chọn mô hình.

## 4. Current Results

Bảng dưới đây được lấy từ `models/metrics.csv` đã lưu trong repository:

| Model | CV MAE | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| Random Forest | **611.95** | 648.82 | 649,599.44 | 805.98 | 0.2857 |
| Gradient Boosting | 619.45 | 653.42 | **625,917.93** | **791.15** | **0.3117** |
| Decision Tree | 688.72 | 743.48 | 960,512.33 | 980.06 | -0.0562 |
| Linear Regression | 788.72 | 671.92 | 698,524.06 | 835.78 | 0.2319 |
| Dummy Mean | 854.40 | 811.34 | 945,125.93 | 972.18 | -0.0393 |

**Mô hình được chọn: Random Forest.** Lý do là Random Forest có CV_MAE thấp nhất (611.95) trong các mô hình đã thử. Cấu hình tốt nhất được lưu trong kết quả tuning hiện tại là `n_estimators=100`, `max_depth=10`, `min_samples_leaf=1`. Trên tập Test, Gradient Boosting có RMSE và R² tốt hơn một chút, trong khi Random Forest có MAE thấp hơn. Đây là khác biệt giữa tiêu chí chọn mô hình bằng cross-validation và các chỉ số đánh giá trên tập Test; cần trình bày minh bạch thay vì chỉ nêu một chỉ số thuận lợi.

R² của mô hình được chọn khoảng 0.286, cho thấy khả năng giải thích biến thiên doanh thu còn ở mức vừa phải. Kết quả phù hợp để minh họa quy trình thực hành và cần được cải thiện trước khi cân nhắc sử dụng trong hoạt động kinh doanh thực tế.

## 5. Project Structure

```text
ML_Lab2/
├── README.md
├── run_pipeline.py                 # Điều phối toàn bộ pipeline
├── demo.py                         # Ví dụ dự đoán cục bộ
├── requirements.txt
├── pytest.ini
│
├── data/
│   ├── raw/
│   │   └── sales_data.csv
│   └── processed/
│       ├── sales_cleaned.csv
│       ├── train_base.csv
│       ├── test_base.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── X_train_fe.csv
│       ├── X_test_fe.csv
│       ├── feature_history.joblib
│       └── preprocessing_summary.json
│
├── doc/
│   ├── description.md
│   ├── report.md                    # Được tạo sau khi chạy pipeline
│   ├── presentation.md
│   ├── audit.md
│   └── algorithm_selection.md
│
├── notebooks/
│   └── practice2_product_sales.ipynb
│
├── figures/
│   ├── eda_sales_distribution.png
│   ├── eda_sales_by_category.png
│   ├── eda_sales_by_country.png
│   ├── eda_monthly_sales_trend.png
│   ├── eda_price_vs_sales.png
│   ├── model_comparison_mae.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
├── models/
│   ├── LinearRegression.joblib
│   ├── DecisionTree.joblib
│   ├── RandomForest.joblib
│   ├── GradientBoosting.joblib
│   ├── DummyMean.joblib
│   ├── best_sales_model.joblib
│   ├── metrics.csv
│   ├── tuning_results.csv
│   ├── predictions.csv
│   ├── feature_importance.csv
│   └── training_summary.json
│
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── training.py
│   ├── eval.py
│   ├── inference.py
│   ├── report.py
│   └── algorithms/                 # Các module thuật toán được giữ trong project
│
└── tests/
    └── test_pipeline.py
```

Một số file trong `data/processed/`, `models/` và `figures/` được sinh ra khi chạy pipeline. Danh sách trên mô tả các output chính, không phải tất cả đều cần tồn tại trước lần chạy đầu tiên.

## 6. Installation

Dự án đang cố định phiên bản thư viện trong `requirements.txt`; repository khuyến nghị Python 3.13 để tương thích với các model artifact đã lưu.

### 6.1. Clone project

```bash
git clone https://github.com/phuonghcv5435-wq/ML_Lab2.git
cd ML_Lab2
```

### 6.2. Create virtual environment

```bash
python3 -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 6.3. Install dependencies

```bash
pip install -r requirements.txt
```

## 7. How to Run

### 7.1. Run the complete pipeline

Đứng tại thư mục gốc của dự án:

```bash
python run_pipeline.py
```

Pipeline lần lượt thực hiện thu thập dữ liệu, EDA, tiền xử lý, tạo đặc trưng, huấn luyện/tuning mô hình, đánh giá và tạo báo cáo tại `doc/report.md`.

### 7.2. Run the notebook

Mở file:

```text
notebooks/practice2_product_sales.ipynb
```

Có thể mở bằng JupyterLab hoặc VS Code Notebook. Chọn đúng môi trường Python đã cài `requirements.txt`, sau đó chạy các cell từ trên xuống.

### 7.3. Run a local prediction

Sau khi đã chạy pipeline và tạo model artifact:

```bash
python demo.py
```

Chương trình tải `models/best_sales_model.joblib`, chuẩn bị feature từ thông tin giao dịch mẫu và in doanh thu dự đoán.

### 7.4. Run tests

```bash
pytest -q
```

Các test hiện có kiểm tra định nghĩa Target, chia tập theo thời gian, tính lịch sử chỉ từ giao dịch trước và việc tạo feature dự đoán mà không làm thay đổi trạng thái lịch sử.

## 8. Important Output Files

| File | Nội dung / mục đích |
|---|---|
| `data/processed/sales_cleaned.csv` | Dữ liệu sau bước làm sạch và tạo Target |
| `data/processed/train_base.csv` | Phần dữ liệu Train trước Feature Engineering |
| `data/processed/test_base.csv` | Phần dữ liệu Test trước Feature Engineering |
| `data/processed/X_train_fe.csv` | Feature đầu vào của tập Train |
| `data/processed/X_test_fe.csv` | Feature đầu vào của tập Test |
| `data/processed/y_train.csv`, `y_test.csv` | Target tương ứng của Train và Test |
| `data/processed/feature_history.joblib` | Trạng thái lịch sử sản phẩm/khách hàng từ Train |
| `models/metrics.csv` | Bảng so sánh chỉ số của các mô hình |
| `models/tuning_results.csv` | Kết quả từng cấu hình tuning |
| `models/predictions.csv` | Giá trị thực tế, dự đoán và sai số tuyệt đối trên Test |
| `models/best_sales_model.joblib` | Bundle chứa mô hình tốt nhất và thông tin cần cho inference |
| `models/training_summary.json` | Tên mô hình được chọn và kết quả tương ứng |
| `models/feature_importance.csv` | Bảng tầm quan trọng đặc trưng của mô hình được lưu |
| `figures/` | Biểu đồ EDA và đánh giá mô hình |
| `doc/report.md` | Báo cáo kết quả được tạo tự động từ output thực tế |

## 9. Testing and Reproducibility

- Dùng `TimeSeriesSplit` trong tập Train để đánh giá và tuning theo thứ tự thời gian.
- Dùng tập Test theo giai đoạn thời gian sau để đánh giá cuối cùng.
- Lưu các chỉ số, tuning results, predictions và model artifact để có thể xem lại kết quả.
- Các phiên bản thư viện được cố định trong `requirements.txt`.

Khi thay đổi dữ liệu hoặc code, cần chạy lại `python run_pipeline.py` để cập nhật các file kết quả. Không nên trộn chỉ số từ những lần chạy khác nhau trong cùng một báo cáo.

## 10. Limitations and Future Improvements

### Limitations

1. Dataset chỉ có 500 giao dịch trong một năm; quy mô nhỏ khiến kết quả có thể thay đổi đáng kể nếu dữ liệu khác.
2. Không có trường quảng cáo, chiến dịch marketing, review/rating hoặc nhãn ngày lễ chính thức.
3. `is_holiday_season` chỉ là proxy cho tháng 11–12, không xác định các ngày lễ cụ thể.
4. Chỉ có một chu kỳ năm nên chưa thể xác nhận mùa vụ lặp lại hằng năm.
5. `quantity` bị loại khỏi X; mô hình không trả lời kịch bản dự đoán doanh thu khi đã biết số lượng dự kiến.
6. Kết quả R² hiện tại của Random Forest khoảng 0.286; chưa đủ cơ sở để xem đây là mô hình production-ready.

### Future Improvements

- Thu thập thêm dữ liệu qua nhiều năm và nhiều nhóm sản phẩm.
- Bổ sung dữ liệu marketing, khuyến mãi, tồn kho hoặc đánh giá sản phẩm nếu có nguồn đáng tin cậy.
- Mở rộng tìm kiếm tham số và đánh giá bằng nhiều giai đoạn hold-out.
- Phân tích sai số theo nhóm sản phẩm, khách hàng và giai đoạn thời gian.
- Thử nghiệm cách đặt Target phù hợp hơn nếu mục tiêu kinh doanh là dự đoán doanh thu theo số lượng đặt trước.

## 11. Documentation

- [Algorithm Selection](doc/algorithm_selection.md) — lý do chọn mô hình và cách đánh giá.
- [Presentation Script](doc/presentation.md) — kịch bản trình bày và demo.
- [Final Audit](doc/audit.md) — kiểm tra phạm vi hoàn thành, kết quả và hạn chế.
- [Project Description](doc/description.md) — mô tả đề tài ban đầu.
- [Practice Notebook](notebooks/practice2_product_sales.ipynb) — notebook đi theo workflow thực hành.

## 12. Author and Repository

- **Project:** Product Sales Prediction — Practice 2
- **Repository:** https://github.com/phuonghcv5435-wq/ML_Lab2
- **Reference project:** https://github.com/ducpp1720-ops/Lab2_Product_Sales
- **Type:** Supervised Learning — Regression
- **Execution:** Local Python project; no API server.

---

*Developed for academic practice in data preprocessing, feature engineering, regression modeling, evaluation, and local inference.*
