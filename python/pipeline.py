import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import matplotlib.pyplot as plt
import seaborn as sns

NUMERIC_FEATURES = [
    'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 
    'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Size'
]

def plot_correlation_heatmap(df_raw):
    fig, ax = plt.subplots(figsize=(16, 10))
    
    numeric_cols = df_raw.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) > 1:
        corr_matrix = df_raw[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
        ax.set_title('Biểu đồ nhiệt Tương quan (Dữ liệu Thô)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, 'Không đủ cột số để vẽ heatmap', horizontalalignment='center', verticalalignment='center')
    
    fig.tight_layout()
    return fig

def step_1_1_handle_missing(df):
    df_step = df.copy()
    markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
    existing_markdown_cols = [col for col in markdown_cols if col in df_step.columns]
    df_step[existing_markdown_cols] = df_step[existing_markdown_cols].fillna(0)
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

def step_1_4_scale_data(df):
    df_step = df.copy()
    
    existing_numeric_features = [col for col in NUMERIC_FEATURES if col in df_step.columns]
    
    scaler = StandardScaler() 
    
    if existing_numeric_features:
        df_step[existing_numeric_features] = scaler.fit_transform(df_step[existing_numeric_features])
    
    return df_step, scaler 


def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "R-squared (R²)": r2,
        "MAE": mae,
        "RMSE": rmse
    }

def run_training_pipeline(X_train, X_test, y_train, y_test):
    results = {}
    models = {} 

    start_time = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    train_time = time.time() - start_time
    results["Linear Regression"] = get_metrics(y_test, y_pred_lr)
    results["Linear Regression"]["Time"] = train_time
    models["Linear Regression"] = lr_model

    start_time = time.time()
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    train_time = time.time() - start_time
    results["Decision Tree"] = get_metrics(y_test, y_pred_dt)
    results["Decision Tree"]["Time"] = train_time
    models["Decision Tree"] = dt_model

    start_time = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    train_time = time.time() - start_time
    results["Random Forest"] = get_metrics(y_test, y_pred_rf)
    results["Random Forest"]["Time"] = train_time
    models["Random Forest"] = rf_model 
    
    return results, models