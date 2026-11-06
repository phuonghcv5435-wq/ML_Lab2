import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_folder = os.path.join(project_root, 'data')
    model_folder = os.path.join(project_root, 'model')
    
    # 1. Đọc dữ liệu kiểm thử (Test)
    X_test_path = os.path.join(data_folder, 'X_test_fe.csv')
    y_test_path = os.path.join(data_folder, 'y_test.csv')
    
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
            # Load mô hình từ file .pkl
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
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    evaluate_models()