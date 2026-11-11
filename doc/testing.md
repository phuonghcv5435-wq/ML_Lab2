# Kiểm thử và xác minh hoạt động (Test and Run Verification)

Dự án có thể được sử dụng hoàn toàn thông qua các câu lệnh Python chạy trực tiếp trên máy tính cá nhân. Dự án không sử dụng máy chủ web (Web Server) và không yêu cầu cài đặt các thư viện phụ thuộc liên quan đến máy chủ web.

## Kết quả kiểm thử (Test Results)

| Hạng mục kiểm tra | Kết quả |
|---|---|
| `python -m compileall -q .` | Thành công (Passed) |
| `python -m pytest -q` | Thành công (Passed - 4 bài kiểm thử) |
| `python run_pipeline.py` | Thực hiện đầy đủ các bước: thu thập dữ liệu, phân tích dữ liệu khám phá (EDA), tiền xử lý dữ liệu, xây dựng đặc trưng, huấn luyện mô hình, đánh giá và tạo báo cáo kết quả. |
| `python demo.py` | Tải mô hình đã lưu và hiển thị giá trị doanh thu dự đoán dưới dạng số. |

## Kết quả kiểm tra lại trên tập dữ liệu (Reproduced Holdout Results)

91 bản ghi cuối cùng theo thứ tự thời gian được giữ lại để làm tập kiểm tra (Test Set).

Mô hình Random Forest được lựa chọn dựa trên tiêu chí có giá trị MAE thấp nhất trong quá trình kiểm định chéo theo chuỗi thời gian (Time-Series Cross-Validation).

### Kết quả đánh giá

| Chỉ số đánh giá | Giá trị |
|---|---:|
| Holdout MAE | 648.820 |
| Holdout MSE | 649,599.435 |
| Holdout RMSE | 805.977 |
| Holdout R² | 0.286 |
| Time-Series CV MAE | 611.947 |

### Lưu ý

Các kết quả trên được tính toán dựa trên tập dữ liệu 500 dòng được cung cấp và các phiên bản thư viện đã được cố định trong dự án (Pinned Package Versions).

Những kết quả này chỉ phản ánh hiệu suất của mô hình trong phạm vi dữ liệu và điều kiện thử nghiệm hiện tại, không đảm bảo mô hình sẽ đạt được hiệu suất tương tự khi triển khai trong môi trường thực tế (Production Environment).