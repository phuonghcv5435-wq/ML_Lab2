# Bảng kiểm tra yêu cầu bài tập — Practice 2: Dự đoán doanh số sản phẩm (Predicting Product Sales)

| Yêu cầu trong PDF | Phần triển khai trong dự án | Trạng thái / Ghi chú |
|---|---|---|
| Xử lý dữ liệu thiếu và giá trị ngoại lệ | `src/data_preprocessing.py`; sử dụng Imputer trong quy trình huấn luyện; xác định ngưỡng IQR trên tập huấn luyện | Đã triển khai. File dữ liệu hiện tại có 500 dòng, không có giá trị thiếu hoặc ngoại lệ theo quy tắc IQR, do đó số lượng bản ghi bị loại bỏ hoặc điều chỉnh là 0. |
| Chuyển đổi các biến phân loại | `OneHotEncoder(handle_unknown="ignore")` trong `src/training.py` | Đã triển khai bên trong Pipeline của mô hình đã lưu. |
| Chia tập huấn luyện và kiểm tra | Chia dữ liệu theo thứ tự thời gian trong `src/data_preprocessing.py` | Đã triển khai: 409 bản ghi huấn luyện và 91 bản ghi kiểm tra có ngày giao dịch muộn hơn. |
| Đánh giá mức độ phổ biến của sản phẩm | Số lượng giao dịch trước đó và doanh thu trung bình trong quá khứ của từng sản phẩm | Đã triển khai bằng các chỉ số đại diện dựa trên dữ liệu hiện có. Tập dữ liệu không có thông tin đánh giá hoặc xếp hạng sản phẩm. |
| Giá trị vòng đời khách hàng (Customer Lifetime Value) | Tổng doanh thu tích lũy từ các giao dịch trước đó của từng khách hàng | Đã triển khai dưới dạng giá trị ước tính dựa trên lịch sử mua hàng hiện có. |
| Phân tích xu hướng mùa vụ | Sử dụng các đặc trưng tháng, quý, thứ trong tuần, ngày cuối tuần và đặc trưng thời gian có tính chu kỳ; kết hợp phân tích EDA theo doanh thu từng tháng | Đã triển khai. Tuy nhiên, dữ liệu chỉ có một năm nên chưa đủ cơ sở để xác nhận đáng tin cậy các quy luật mùa vụ lặp lại hằng năm. |
| Lựa chọn mô hình hồi quy | Linear Regression, Decision Tree, Random Forest, Gradient Boosting và mô hình cơ sở Dummy | Đã triển khai và so sánh các mô hình. |
| Tối ưu tham số mô hình (Hyperparameter Tuning) | `GridSearchCV` kết hợp `TimeSeriesSplit` | Đã triển khai cho các mô hình dựa trên cây quyết định. |
| Huấn luyện mô hình | `src/training.py` | Đã triển khai. Các Pipeline bao gồm bước tiền xử lý dữ liệu và mô hình đã huấn luyện được lưu trong thư mục `models/`. |
| Đánh giá mô hình | MAE, MSE, RMSE và R² trên tập kiểm tra có ngày giao dịch muộn hơn | Đã triển khai. Kết quả được lưu tại `models/metrics.csv` và `doc/report.md`. |
| Dự đoán trên máy tính cá nhân (Local Prediction) | `models/best_sales_model.joblib`, `src/inference.py`, `demo.py` | Đã triển khai. Pipeline đã lưu có thể trực tiếp dự đoán doanh thu cho một giao dịch mới thông qua Python. |
| ARIMA / Prophet (Tùy chọn) | Xây dựng chuỗi doanh thu theo tháng và biểu đồ phân tích | Đã xem xét nhưng chưa huấn luyện mô hình. Dữ liệu chỉ có 12 điểm theo tháng (một chu kỳ năm), chưa đủ để kiểm chứng đáng tin cậy các quy luật mùa vụ lặp lại. |
| Hệ thống gợi ý sản phẩm (Tùy chọn) | Chưa triển khai | Đây là nội dung được đề cập như một phần mở rộng trong bài tập. Dữ liệu hiện tại không có điểm đánh giá hoặc thông tin phản hồi rõ ràng về sở thích của khách hàng. |

## Lựa chọn biến mục tiêu và ngăn chặn rò rỉ dữ liệu (Target Choice and Leakage Prevention)

Nhóm lựa chọn doanh thu của từng giao dịch (`sales`) làm biến mục tiêu, được xác định theo công thức:

`sales = unit_price * quantity`

Do thuộc tính `quantity` được sử dụng trực tiếp để tính biến mục tiêu nên nhóm chủ động loại bỏ thuộc tính này khỏi tập đặc trưng đầu vào của mô hình.

Điều này có nghĩa là mô hình được huấn luyện để ước tính doanh thu của một giao dịch mà không được cung cấp số lượng sản phẩm thực tế của chính giao dịch đó.

Cách thực hiện này giúp hạn chế hiện tượng rò rỉ dữ liệu mục tiêu (Target Leakage), đảm bảo mô hình không sử dụng trực tiếp một thành phần cấu thành biến cần dự đoán.

## Ghi chú về repository tham khảo (Reference Repository Note)

Nhóm đã kiểm tra file ZIP được cung cấp và sử dụng nội dung bên trong làm dự án khởi đầu (Starter Project).

Đường dẫn repository mẫu trên mạng được đề cập trong quá trình thực hiện không thể truy cập từ môi trường chạy hiện tại. Vì vậy, dự án không khẳng định đã thực hiện đối chiếu chi tiết từng dòng mã nguồn với repository trực tuyến.

Việc triển khai dự án được xây dựng dựa trên nội dung của file PDF đề bài được cung cấp, kết hợp với mã nguồn và dữ liệu có trong file ZIP đã tải lên.