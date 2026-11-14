# Kịch bản trình bày — Practice 2: Product Sales Prediction

**Thời lượng gợi ý:** 2–3 phút, chưa tính phần giảng viên đặt câu hỏi.

## 1. Script trình bày

“Đề tài của em là **Practice 2 — Product Sales Prediction**, xây dựng mô hình Machine Learning để dự đoán doanh thu của một giao dịch sản phẩm.

Đây là bài toán Supervised Learning dạng Regression vì biến mục tiêu `sales` là một giá trị số liên tục. Em xác định doanh thu bằng công thức `sales = unit_price × quantity`. Tuy nhiên, `quantity` chỉ dùng để tính Target và được loại khỏi Feature đầu vào cuối cùng, nhằm tránh để mô hình học lại trực tiếp công thức tạo Target. Vì vậy mô hình hiện tại không dùng để tính doanh thu theo một số lượng đặt trước.

Dataset của em có 500 giao dịch với thông tin sản phẩm, khách hàng, đơn giá, số lượng, ngày giao dịch và quốc gia. Sau bước kiểm tra và làm sạch, dữ liệu còn 500 dòng. Em chia dữ liệu theo thời gian: 409 dòng đầu cho Train và 91 dòng ở giai đoạn sau cho Test. Cách chia này giúp mô phỏng việc dùng dữ liệu quá khứ để dự đoán giao dịch ở thời điểm sau.

Tiếp theo, em tạo các đặc trưng lịch sử như mức độ phổ biến trước đó của sản phẩm, số lần mua trước của khách hàng, doanh thu tích lũy trước đó của khách hàng và các đặc trưng thời gian như tháng, quý, ngày trong tuần và cuối tuần. Những đặc trưng lịch sử chỉ sử dụng giao dịch đã xảy ra trước giao dịch đang xét.

Em thử bốn mô hình: Linear Regression, Decision Tree, Random Forest và Gradient Boosting. Ngoài ra, em dùng Dummy Regressor làm baseline. Các mô hình cây được tuning bằng GridSearchCV kết hợp TimeSeriesSplit. Tiêu chí chọn mô hình là CV MAE thấp nhất trên tập Train.

Trong kết quả hiện tại, Random Forest được chọn vì có CV MAE thấp nhất, khoảng **611.95**. Trên tập Test, mô hình đạt MAE khoảng **648.82**, RMSE khoảng **805.98** và R² khoảng **0.286**. Gradient Boosting có RMSE Test thấp hơn, vì vậy em trình bày cả hai kết quả để giải thích rằng mô hình được chọn theo CV MAE chứ không chọn theo một chỉ số Test riêng lẻ.

Cuối cùng, mô hình tốt nhất cùng lịch sử giao dịch được lưu trong `models/best_sales_model.joblib`. Em có thể chạy `python demo.py` để tải mô hình và dự đoán một giao dịch mẫu ngay trên máy tính cá nhân. Dự án không sử dụng API hay máy chủ web.”

## 2. Demo Sequence

Nên thực hiện theo thứ tự sau để người nghe thấy rõ từ dữ liệu đến kết quả:

1. Mở `README.md`, giới thiệu mục tiêu, Target và tổng quan quy trình.
2. Mở `notebooks/practice2_product_sales.ipynb`, chỉ vào các phần Load Data, EDA, Preprocessing, Feature Engineering và Model Training.
3. Mở `data/processed/preprocessing_summary.json`, giải thích 500 dòng dữ liệu, 409 Train và 91 Test; Train kết thúc ngày 12/10/2022, Test bắt đầu ngày 13/10/2022.
4. Mở `models/metrics.csv`, so sánh năm dòng kết quả gồm bốn mô hình hồi quy và Dummy Mean.
5. Mở `models/tuning_results.csv`, chỉ ra một số cấu hình tham số đã được thử nghiệm.
6. Mở `figures/model_comparison_mae.png` và `figures/actual_vs_predicted.png` để giải thích kết quả.
7. Chạy `python demo.py` để minh họa dự đoán cục bộ bằng model đã lưu.
8. Chạy `pytest -q` để thực hiện các kiểm thử của dự án.

## 3. Câu hỏi giảng viên có thể hỏi

### Vì sao đây là Regression?

Vì Target `sales` là số tiền/doanh thu dạng số liên tục. Mô hình cần dự đoán một giá trị số, không phải phân loại giao dịch thành các nhóm.

### Tại sao không đưa `quantity` vào X?

Vì `sales` được tạo trực tiếp từ `unit_price × quantity`. Nếu đưa lại `quantity` vào X, mô hình có thể khai thác trực tiếp một thành phần của công thức tạo Target. Đây là quyết định thiết kế của bài thực hành. Đổi lại, mô hình hiện tại không trả lời câu hỏi dự đoán tổng doanh thu khi đã biết số lượng đặt trước.

### Vì sao chia Train/Test theo thời gian?

Vì dữ liệu có ngày giao dịch. Dùng các giao dịch sớm hơn để huấn luyện và giai đoạn sau để kiểm tra giúp mô phỏng tình huống dự đoán trong tương lai tốt hơn so với chia ngẫu nhiên.

### Product Popularity và Customer Lifetime Value được tính như thế nào?

Product Popularity được ước tính bằng số giao dịch trước đó của sản phẩm. Customer Lifetime Value trong project là tổng doanh thu tích lũy từ các giao dịch trước đó của khách hàng. Cả hai chỉ dùng lịch sử có trước giao dịch hiện tại.

### Vì sao Random Forest được chọn dù Gradient Boosting có RMSE Test thấp hơn?

Vì quy tắc chọn mô hình được xác định trước là CV MAE thấp nhất trên tập Train. Random Forest có CV MAE 611.95, thấp hơn Gradient Boosting 619.45. Test chỉ dùng để đánh giá cuối cùng, không dùng để chọn mô hình.

### R² khoảng 0.286 nói lên điều gì?

Mô hình giải thích được một phần biến thiên của doanh thu nhưng hiệu quả còn hạn chế. Đây là kết quả thực hành trên bộ dữ liệu 500 giao dịch, không phải bằng chứng về độ chính xác khi triển khai kinh doanh.

### Tại sao không có dữ liệu quảng cáo hoặc ngày lễ?

Dataset hiện không có các trường chi phí quảng cáo, chiến dịch marketing hay nhãn ngày lễ chính thức. Project không tự tạo các dữ liệu đó. Cờ tháng 11–12 chỉ là proxy mùa cao điểm, không phải dữ liệu ngày lễ xác thực.

### Có API hoặc ứng dụng web không?

Không. Dự án được giữ ở dạng chương trình Python chạy cục bộ. `demo.py` gọi hàm inference để tải model và dự đoán trực tiếp.
