DỰ ĐOÁN DOANH SỐ SẢN PHẨM (PRODUCT SALES PREDICTION)
1. WHAT – Chúng ta sẽ làm gì?

Dự án này tập trung xây dựng một mô hình dự đoán doanh số bán sản phẩm bằng cách sử dụng các kỹ thuật Học máy (Machine Learning).

Dữ liệu thực tế đi kèm dự án có 500 giao dịch từ ngày 01/01/2022 đến 31/12/2022, với các trường:

Thông tin sản phẩm: mã sản phẩm, tên sản phẩm, danh mục và đơn giá.
Thông tin khách hàng: mã khách hàng, độ tuổi, giới tính và quốc gia.
Giao dịch: ngày đặt hàng và số lượng mua (chỉ dùng để tính biến mục tiêu).
Thông tin thời gian: tháng, quý, thứ trong tuần và mùa mua sắm được suy ra từ ngày giao dịch.

Đề bài có nêu marketing campaign/ad spend, review/rating và holiday information, nhưng các trường này không có trong file dữ liệu hiện tại nên không được tự tạo hoặc giả định.
Từ những dữ liệu này, hệ thống sẽ học được các yếu tố ảnh hưởng đến doanh số và đưa ra dự đoán về doanh thu giao dịch của sản phẩm (sales).

Mục tiêu chính: Xây dựng một mô hình có khả năng dự đoán doanh số sản phẩm dựa trên các dữ liệu đầu vào.

Đây là bài toán Supervised Learning (Học có giám sát) vì mô hình được huấn luyện bằng dữ liệu lịch sử đã có kết quả doanh số thực tế.
Cụ thể, dự án thuộc nhóm Regression (Hồi quy) vì mục tiêu là dự đoán một giá trị số liên tục là doanh thu giao dịch (`sales`), thay vì phân loại dữ liệu thành các nhóm.

Input(X): Đơn giá, thông tin sản phẩm/khách hàng, lịch sử giao dịch trước đó và đặc trưng thời gian; không dùng `quantity` vì đây là thành phần tính target.
Output(Y): Doanh thu giao dịch (`sales = unit_price × quantity`)
Goal: Dự đoán doanh thu giao dịch sản phẩm (`sales`)

2. WHY – Tại sao cần thực hiện dự án này?

Trong kinh doanh thương mại điện tử, việc dự đoán doanh số đóng vai trò quan trọng vì doanh nghiệp thường gặp khó khăn trong việc xác định nhu cầu mua sắm của khách hàng.

Nếu không dự đoán được doanh số, doanh nghiệp có thể gặp phải những vấn đề như:

Nhập quá nhiều hàng dẫn đến tình trạng tồn kho.
Nhập quá ít hàng khiến sản phẩm bị thiếu khi khách hàng có nhu cầu.
Sử dụng ngân sách quảng cáo chưa hiệu quả.
Khó khăn trong việc lập kế hoạch kinh doanh.
Vì vậy, dự án được thực hiện nhằm:

Hỗ trợ doanh nghiệp quản lý hàng tồn kho hiệu quả hơn.
Giảm thiểu tình trạng thiếu hàng hoặc dư thừa hàng hóa.
Hỗ trợ lập kế hoạch quảng cáo và tiếp thị.
Phân tích những yếu tố ảnh hưởng đến doanh số.
Hỗ trợ doanh nghiệp đưa ra quyết định kinh doanh dựa trên dữ liệu.

So với phương pháp dự đoán thủ công dựa trên kinh nghiệm, Machine Learning có khả năng phân tích lượng lớn dữ liệu và tìm ra các mối quan hệ giữa những yếu tố ảnh hưởng đến doanh số.

Thông qua quá trình huấn luyện, mô hình có thể:
Nhận biết xu hướng mua sắm của khách hàng.
Phân tích mức độ ảnh hưởng của giá bán và chương trình khuyến mãi.
Phát hiện các yếu tố có mối quan hệ với sự thay đổi doanh số.
Đưa ra dự đoán dựa trên dữ liệu thay vì chỉ dựa vào cảm tính.

3. WHEN – Khi nào cần sử dụng mô hình?

Mô hình dự đoán doanh số có thể được sử dụng trong nhiều tình huống kinh doanh khác nhau, chẳng hạn như:

Trước khi nhập hàng: Doanh nghiệp dự đoán nhu cầu mua sắm để xác định số lượng sản phẩm cần nhập.
Trước các dịp khuyến mãi: Dự đoán doanh số trong những chương trình giảm giá hoặc ngày hội mua sắm.
Vào các thời điểm khác nhau trong năm: Phân tích sự thay đổi doanh số theo mùa, ngày lễ và các dịp đặc biệt.
Khi lập kế hoạch kinh doanh: Dự đoán doanh số để hỗ trợ xây dựng kế hoạch bán hàng và phân bổ ngân sách.
Khi ra mắt sản phẩm mới: Sử dụng những thông tin và dữ liệu liên quan để hỗ trợ ước tính nhu cầu thị trường nếu có đủ dữ liệu phù hợp.
Nhìn chung, mô hình có thể được sử dụng bất cứ khi nào doanh nghiệp cần ước tính doanh số trong tương lai để chuẩn bị kế hoạch kinh doanh.

Mô hình hướng đến việc hỗ trợ các đối tượng trong hoạt động kinh doanh như:
Nhà quản lý: Theo dõi và dự đoán tình hình kinh doanh.
Nhân viên quản lý kho: Ước tính nhu cầu hàng hóa để lập kế hoạch nhập hàng.
Bộ phận Marketing: Đánh giá và lập kế hoạch cho các chiến dịch quảng cáo.
Chủ cửa hàng: Xác định những sản phẩm có nhu cầu cao để xây dựng chiến lược bán hàng.

4. HOW – Dự án sẽ được thực hiện như thế nào?

Dự án sẽ được triển khai thông qua các bước chính sau:

Bước 1: Thu thập dữ liệu (Data Collection)

Nạp và kiểm tra bộ dữ liệu CSV đã được cung cấp trong `data/raw/sales_data.csv`. Tài liệu `data/README.md` ghi lại nguồn dữ liệu và các cột thực tế. Dữ liệu hiện tại không có trường chi phí quảng cáo hoặc thông tin rating/review.

Ngoài việc tìm kiếm dữ liệu trên Kaggle, nhóm cần xác định các tiêu chí lựa chọn dataset:
Dữ liệu có nguồn gốc rõ ràng.
Có thể xác định biến mục tiêu doanh thu giao dịch (`sales`).
Có các thuộc tính liên quan đến sản phẩm, giá bán và thời gian.
Có đủ số lượng bản ghi để phục vụ huấn luyện và kiểm tra mô hình.

Bước 2: Tiền xử lý dữ liệu (Data Preparation)

Kiểm tra và xử lý dữ liệu bị thiếu.
Loại bỏ các bản ghi trùng lặp.
Phát hiện và xử lý các giá trị bất thường.
Chuyển đổi dữ liệu dạng chữ thành dạng số khi cần thiết.
Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
Bước 3: Tạo đặc trưng (Feature Engineering)

Xây dựng thêm các đặc trưng có thể hỗ trợ quá trình dự đoán, chẳng hạn như:

Mức độ phổ biến của sản phẩm.
Tần suất mua hàng của khách hàng.
Xu hướng doanh số theo mùa.
Tỷ lệ chi phí quảng cáo (không triển khai được vì dataset hiện tại không có ad spend).
Bước 4: Lựa chọn mô hình (Model Selection)

Thử nghiệm một số thuật toán Học máy phù hợp với bài toán dự đoán giá trị số:

Linear Regression.
Decision Tree Regressor.
Random Forest Regressor.
Gradient Boosting Regressor.
Bước 5: Huấn luyện mô hình (Model Training)

Sử dụng tập dữ liệu huấn luyện để mô hình học mối quan hệ giữa các đặc trưng đầu vào và doanh số sản phẩm.

Bước 6: Đánh giá mô hình (Model Evaluation)

Sử dụng các chỉ số để đánh giá độ chính xác của mô hình:

MSE (Mean Squared Error).
MAE (Mean Absolute Error).
RMSE (Root Mean Squared Error).
So sánh kết quả giữa các mô hình để lựa chọn mô hình phù hợp nhất.

Sử dụng các biểu đồ để thể hiện kết quả phân tích và đánh giá mô hình:
Biểu đồ phân phối doanh thu giao dịch (`sales`).
Biểu đồ thể hiện mối quan hệ giữa đơn giá và doanh thu giao dịch.
Biểu đồ so sánh doanh số thực tế và doanh số dự đoán.
Biểu đồ so sánh hiệu suất giữa các mô hình.
Biểu đồ thể hiện mức độ quan trọng của các đặc trưng (Feature Importance) đối với những mô hình hỗ trợ.

Bước 7: Lưu và sử dụng mô hình (Model Saving and Inference)

Lưu pipeline tiền xử lý và mô hình đã huấn luyện để dự đoán doanh thu giao dịch mới. Người dùng chạy dự đoán trực tiếp bằng Python thông qua chương trình `demo.py`.

5. Kết quả mong đợi (Expected Results)

Sau khi hoàn thành dự án, chúng tôi kỳ vọng:

Xây dựng được mô hình dự đoán doanh số sản phẩm.
Xác định được những yếu tố ảnh hưởng đến doanh số.
So sánh và lựa chọn được thuật toán phù hợp.
Đánh giá được hiệu suất dự đoán của mô hình.
Hỗ trợ doanh nghiệp quản lý hàng tồn kho và lập kế hoạch kinh doanh hiệu quả hơn.
6. Kết luận (Conclusion)

Dự án Dự đoán doanh số sản phẩm ứng dụng Machine Learning nhằm giải quyết vấn đề dự đoán nhu cầu mua sắm trong lĩnh vực thương mại điện tử.

Thông qua việc phân tích dữ liệu lịch sử và xây dựng mô hình dự đoán, dự án hướng đến việc cung cấp thông tin hữu ích giúp doanh nghiệp tối ưu hóa hoạt động kinh doanh và đưa ra quyết định chính xác hơn.

7. Ghi chú về biến mục tiêu và giới hạn dữ liệu

Trong phiên bản hoàn thiện này, biến mục tiêu được thống nhất là `sales = unit_price × quantity` (doanh thu của một giao dịch). Vì `quantity` là thành phần dùng để tính target, cột này được loại khỏi đầu vào mô hình để tránh rò rỉ mục tiêu trực tiếp. Bộ dữ liệu hiện có không cung cấp số lượt đánh giá, rating, chi phí quảng cáo, chiến dịch marketing hoặc cờ ngày lễ; do đó nhóm sử dụng số giao dịch trước đây làm proxy cho độ phổ biến sản phẩm và doanh thu tích lũy trước đây làm proxy cho customer lifetime value. Những giới hạn này được trình bày trong báo cáo kết quả.
