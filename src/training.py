import os
import sys
import pandas as pd
import joblib

# 1. Thêm thư mục gốc dự án vào sys.path để import các module từ src/algorithms/
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
project_root = os.path.dirname(current_dir)            # ML_Lab2/
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. Import hàm get_model từ các thuật toán trong src/algorithms/
from src.algorithms.linear_regression import get_model as get_lr
from src.algorithms.decision_tree import get_model as get_dt
from src.algorithms.random_forest import get_model as get_rf
from src.algorithms.gradient_boosting import get_model as get_gb

# SỬA: Đọc dữ liệu từ 'data/processed' và lưu mô hình vào 'models'
def train_models(data_dir='data/processed', save_model_dir='models'):
    data_folder = os.path.join(project_root, data_dir)
    save_model_folder = os.path.join(project_root, save_model_dir)
    
    if not os.path.exists(save_model_folder):
        os.makedirs(save_model_folder)

    x_train_path = os.path.join(data_folder, 'X_train_fe.csv')
    y_train_path = os.path.join(data_folder, 'y_train.csv')

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        print("❌ Không tìm thấy dữ liệu! Hãy chắc chắn đã chạy data_preprocessing.py và feature_engineering.py.")
        return

    print("🤖 Đang tải dữ liệu và huấn luyện từ các module trong src/algorithms/...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Tạo danh sách các mô hình lấy từ thư mục src/algorithms
    models = {
        'LinearRegression': get_lr(),
        'DecisionTree': get_dt(),
        'RandomForest': get_rf(),
        'GradientBoosting': get_gb()
    }

    for name, model in models.items():
        print(f"  ⚡ Đang huấn luyện: {name}...")
        model.fit(X_train, y_train)
        
        # Lưu file binary .pkl tại thư mục models/
        model_filepath = os.path.join(save_model_folder, f'{name}.pkl')
        joblib.dump(model, model_filepath)
        print(f"  💾 Đã xuất mô hình: {model_filepath}")

    print("\n🎉 Hoàn thành huấn luyện! Tất cả các file .pkl đã được lưu tại thư mục models/")

if __name__ == "__main__":
    train_models()