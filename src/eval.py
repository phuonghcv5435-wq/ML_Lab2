import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# SỬA: Đọc dữ liệu từ 'data/processed' và load mô hình từ 'models'
def evaluate_models(data_dir='data/processed', model_dir='models'):
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_dir)            # ML_Lab2/
    
    data_folder = os.path.join(project_root, data_dir)
    model_folder = os.path.join(project_root, model_dir)
    
    # 1. Đọc dữ liệu kiểm thử (Test) từ data/processed/
    X_test_path = os.path.join(data_folder, 'X_test_fe.csv')
    y_test_path = os.path.join(data_folder, 'y_test.csv')
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"❌ Không tìm thấy dữ liệu kiểm thử tại: {data_folder}")
        print("💡 Hãy chạy 'python3 src/feature_engineering.py' trước!")
        return

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    model_names = ['LinearRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting']
    
    print("\n" + "="*60)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TẬP TEST")
    print("="*60)
    
    results = []
    
    for name in model_names:
        model_path = os.path.join(model_folder, f'{name}.pkl')
        if os.path.exists(model_path):
            # Load mô hình từ file .pkl trong models/
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Mô hình': name,
                'MAE ($)': round(mae, 2),
                'MSE': round(mse, 2),
                'R2 Score': round(r2, 4)
            })
        else:
            print(f"⚠️ Không tìm thấy file mô hình tại: {model_path}")

    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        print("="*60)

if __name__ == "__main__":
    evaluate_models()