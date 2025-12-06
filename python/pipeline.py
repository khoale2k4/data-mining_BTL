# pipeline.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import time

NUMERIC_FEATURES = [
    'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 
    'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Size'
]

def plot_correlation_heatmap(df_raw):
    """Vẽ biểu đồ nhiệt tương quan"""
    fig, ax = plt.subplots(figsize=(16, 10))
    numeric_df = df_raw.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
        ax.set_title('Biểu đồ nhiệt Tương quan (Correlation Heatmap)')
    else:
        ax.text(0.5, 0.5, 'Không đủ cột số để vẽ heatmap', ha='center')
    
    fig.tight_layout()
    return fig

def plot_sales_over_time(df):
    """Vẽ xu hướng doanh số theo thời gian"""
    df_plot = df.copy()
    if 'Date' in df_plot.columns:
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
    
    sales_over_time = df_plot.groupby('Date')['Weekly_Sales'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(data=sales_over_time, x='Date', y='Weekly_Sales', ax=ax, color='#1f77b4')
    ax.set_title('Xu hướng Doanh số trung bình theo Thời gian')
    ax.set_ylabel('Doanh số Trung bình')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_actual_vs_predicted(model, X_test, y_test, model_name="Model"):
    """Vẽ biểu đồ Thực tế vs Dự đoán"""
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if len(y_test) > 2000:
        indices = np.random.choice(len(y_test), 2000, replace=False)
        y_test_plot = y_test.iloc[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_test_plot = y_test
        y_pred_plot = y_pred

    ax.scatter(y_test_plot, y_pred_plot, alpha=0.3, color='blue')
    
    p1 = max(max(y_pred_plot), max(y_test_plot))
    p2 = min(min(y_pred_plot), min(y_test_plot))
    ax.plot([p1, p2], [p1, p2], 'r--', lw=2)
    
    ax.set_title(f'Thực tế vs. Dự đoán ({model_name})')
    ax.set_xlabel('Thực tế')
    ax.set_ylabel('Dự đoán')
    fig.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    """Vẽ Feature Importance"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 15
    top_indices = indices[:top_n]
    y_labels = [feature_names[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=importances[top_indices], y=y_labels, hue=y_labels, legend=False, ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Đặc trưng quan trọng nhất')
    fig.tight_layout()
    return fig

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """Vẽ đường cong học tập (Learning Curve)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(X) > 50000:
        indices = np.random.choice(len(X), 20000, replace=False)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
    else:
        X_sample, y_sample = X, y

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_sample, y_sample, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='neg_root_mean_squared_error'
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    ax.set_title(title)
    ax.set_xlabel("Số lượng mẫu huấn luyện")
    ax.set_ylabel("RMSE (Thấp hơn là tốt hơn)")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    return fig

def step_1_1_handle_missing(df):
    df_step = df.copy()
    markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
    existing = [c for c in markdown_cols if c in df_step.columns]
    df_step[existing] = df_step[existing].fillna(0)
    return df_step

def step_1_2_handle_noise(df):
    df_step = df.copy()
    if 'Weekly_Sales' in df_step.columns:
        df_step.loc[df_step['Weekly_Sales'] < 0, 'Weekly_Sales'] = 0
    return df_step

def step_1_3_feature_engineering(df):
    df_step = df.copy()
    if 'Date' in df_step.columns:
        df_step['Date'] = pd.to_datetime(df_step['Date'])
        df_step['Year'] = df_step['Date'].dt.year
        df_step['Month'] = df_step['Date'].dt.month
        df_step['WeekOfYear'] = df_step['Date'].dt.isocalendar().week.astype(int)
        df_step['Day'] = df_step['Date'].dt.day
        df_step = df_step.drop('Date', axis=1)

    if 'IsHoliday' in df_step.columns:
        df_step['IsHoliday'] = df_step['IsHoliday'].astype(int)

    if 'Type' in df_step.columns:
        df_step = pd.get_dummies(df_step, columns=['Type'], prefix='Type', drop_first=False, dtype=int)
    return df_step

def step_1_4_prepare_scaler(df):
    """Chỉ khởi tạo scaler, chưa biến đổi dữ liệu ngay"""
    df_step = df.copy()
    scaler = StandardScaler() 
    return df_step, scaler 

def apply_scaling(X_train, X_test, scaler):
    """Fit scaler trên Train, sau đó Transform cả Train và Test"""
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    cols_to_scale = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    
    if cols_to_scale:
        scaler.fit(X_train[cols_to_scale])
        
        X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        
    return X_train_scaled, X_test_scaled, scaler

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"R-squared (R²)": r2, "MAE": mae, "RMSE": rmse}

def run_training_pipeline(X_train, X_test, y_train, y_test, params=None):
    results = {}
    models = {} 
    
    if params is None:
        params = {
            'dt_max_depth': None,
            'rf_n_estimators': 50,
            'rf_max_depth': None
        }

    start = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["Linear Regression"] = get_metrics(y_test, lr.predict(X_test))
    results["Linear Regression"]["Time"] = time.time() - start
    models["Linear Regression"] = lr

    start = time.time()
    dt = DecisionTreeRegressor(
        random_state=42, 
        max_depth=params.get('dt_max_depth', None)
    )
    dt.fit(X_train, y_train)
    results["Decision Tree"] = get_metrics(y_test, dt.predict(X_test))
    results["Decision Tree"]["Time"] = time.time() - start
    models["Decision Tree"] = dt

    start = time.time()
    rf = RandomForestRegressor(
        n_estimators=params.get('rf_n_estimators', 50), 
        max_depth=params.get('rf_max_depth', None),
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["Random Forest"] = get_metrics(y_test, rf.predict(X_test))
    results["Random Forest"]["Time"] = time.time() - start
    models["Random Forest"] = rf 
    
    return results, models

def save_artifacts(model, scaler, model_name="best_model"):
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, f'models/{model_name}.pkl')
    joblib.dump(scaler, f'models/{model_name}_scaler.pkl')

def load_artifacts(model_name="best_model"):
    m_path = f'models/{model_name}.pkl'
    s_path = f'models/{model_name}_scaler.pkl'
    if os.path.exists(m_path) and os.path.exists(s_path):
        return joblib.load(m_path), joblib.load(s_path)
    return None, None