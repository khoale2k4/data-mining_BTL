# full.py
import time
from matplotlib import pyplot as plt
import pandas as pd
import os
import pipeline 
from sklearn.model_selection import train_test_split

INPUT_FILE = '../data/walmart_recruiting_dataset.csv'
OUTPUT_IMAGE_DIR = '../output_images'
OUTPUT_MODEL_DIR = '../models'

MODEL_FILES = {
    "Linear Regression": "linear_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest"
}

def main():
    print("="*50)
    print("BẮT ĐẦU PIPELINE TỰ ĐỘNG (ALL MODELS)")
    print("="*50)

    if not os.path.exists(INPUT_FILE):
        print(f"[LỖI] Không tìm thấy file '{INPUT_FILE}'.")
        return

    print(f"[1/5] Đang tải dữ liệu từ '{INPUT_FILE}'...")
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"   -> Đã tải {len(df_raw)} dòng dữ liệu.")

    print("\n[2/5] Đang thực hiện Tiền xử lý dữ liệu...")
    df_step1 = pipeline.step_1_1_handle_missing(df_raw)
    df_step2 = pipeline.step_1_2_handle_noise(df_step1)
    df_step3 = pipeline.step_1_3_feature_engineering(df_step2)
    
    print("   -> Chuẩn bị Scaler (chưa transform)...")
    df_processed, scaler_obj = pipeline.step_1_4_prepare_scaler(df_step3)
    
    print(f"   -> Tiền xử lý thô hoàn tất.")

    print("\n[3/5] Đang chia tập Huấn luyện / Kiểm tra (Split)...")
    y = df_processed['Weekly_Sales']
    X = df_processed.drop('Weekly_Sales', axis=1)
    
    feature_names = X.columns.tolist()
    
    split_ratio = 0.8
    split_index = int(len(df_processed) * split_ratio)

    X_train_raw, X_test_raw = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("   -> Đang chuẩn hóa dữ liệu (Fit on Train, Transform All)...")
    X_train, X_test, fitted_scaler = pipeline.apply_scaling(X_train_raw, X_test_raw, scaler_obj)

    print("\n[4/5] Kiểm tra/Huấn luyện Models...")
    
    results = {}
    models = {}
    
    all_models_exist = True
    for model_name, file_name in MODEL_FILES.items():
        m, s = pipeline.load_artifacts(file_name)
        if m is None:
            all_models_exist = False
            break
    
    if all_models_exist:
        print("   -> Đã tìm thấy model cũ. Đang tải...")
        for model_name, file_name in MODEL_FILES.items():
            model, _ = pipeline.load_artifacts(file_name)
            models[model_name] = model
            
            y_pred = model.predict(X_test)
            metrics = pipeline.get_metrics(y_test, y_pred)
            metrics['Time'] = 0.0 
            results[model_name] = metrics
    else:
        print("   -> Huấn luyện mới từ đầu...")
        results, models = pipeline.run_training_pipeline(X_train, X_test, y_train, y_test)
        
        print("\n   -> Đang lưu TẤT CẢ model...")
        for model_name, model in models.items():
            file_name = MODEL_FILES.get(model_name, model_name.replace(" ", "_").lower())
            pipeline.save_artifacts(model, fitted_scaler, model_name=file_name)
            print(f"      + Đã lưu: {file_name}.pkl")

    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ CHI TIẾT")
    print(f"{'Mô hình':<20} | {'RMSE':<15}  | {'MAE':<15} | {'R²':<10}")
    print("-" * 75)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} | ${metrics['RMSE']:,.2f}".ljust(39) + f" | ${metrics['MAE']:,.2f}".ljust(17) + f"  | {metrics['R-squared (R²)']:.4f}")

    print(f"\n[6/6] Đang vẽ biểu đồ phân tích vào '{OUTPUT_IMAGE_DIR}'...")
    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)
        
    try:
        print("   [Chung] Vẽ Sales Trend...")
        fig1 = pipeline.plot_sales_over_time(df_raw)
        fig1.savefig(os.path.join(OUTPUT_IMAGE_DIR, '0_sales_seasonality.png'))
        plt.close(fig1)

        for model_name, model in models.items():
            print(f"   [{model_name}] Đang vẽ biểu đồ đánh giá...")
            safe_name = MODEL_FILES.get(model_name, model_name.replace(" ", "_").lower())
            
            fig_act = pipeline.plot_actual_vs_predicted(model, X_test, y_test, model_name)
            fig_act.savefig(os.path.join(OUTPUT_IMAGE_DIR, f'1_actual_vs_predicted_{safe_name}.png'))
            plt.close(fig_act) 

            if hasattr(model, 'feature_importances_'):
                fig_imp = pipeline.plot_feature_importance(model, feature_names)
                if fig_imp:
                    fig_imp.savefig(os.path.join(OUTPUT_IMAGE_DIR, f'3_feature_importance_{safe_name}.png'))
                    plt.close(fig_imp)
            
            if model_name == "Random Forest": 
                print(f"      -> Vẽ Learning Curve cho {model_name}...")
                fig_lc = pipeline.plot_learning_curve(model, X_train, y_train, title=f"Học tập: {model_name}")
                fig_lc.savefig(os.path.join(OUTPUT_IMAGE_DIR, f'4_learning_curve_{safe_name}.png'))
                plt.close(fig_lc)

        print("\nTẤT CẢ BIỂU ĐỒ & MODEL ĐÃ ĐƯỢC LƯU THÀNH CÔNG!")

    except Exception as e:
        print(f"\n[LỖI VẼ BIỂU ĐỒ]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()