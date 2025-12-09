import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import pipeline 
from pipeline import NUMERIC_FEATURES 

if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.df_raw = None
    st.session_state.df_processed = None
    st.session_state.scaler = None
    st.session_state.results = None
    st.session_state.models = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.X_columns = None

def go_to_step_2_eda(): st.session_state.step = 2

def run_step_1_1():
    st.session_state.df_step_1_1 = pipeline.step_1_1_handle_missing(st.session_state.df_raw)
    st.session_state.step = 3

def run_step_1_2():
    st.session_state.df_step_1_2 = pipeline.step_1_2_handle_noise(st.session_state.df_step_1_1)
    st.session_state.step = 4

def run_step_1_3():
    st.session_state.df_step_1_3 = pipeline.step_1_3_feature_engineering(st.session_state.df_step_1_2)
    st.session_state.step = 5

def run_step_1_4():
    df_proc, scaler = pipeline.step_1_4_prepare_scaler(st.session_state.df_step_1_3)
    st.session_state.df_processed = df_proc
    st.session_state.scaler = scaler
    st.session_state.step = 6

def run_step_3_split_and_scale():
    df = st.session_state.df_processed
    y = df['Weekly_Sales']
    X = df.drop('Weekly_Sales', axis=1)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled, X_test_scaled, fitted_scaler = pipeline.apply_scaling(X_train_raw, X_test_raw, st.session_state.scaler)
    
    st.session_state.X_train = X_train_scaled
    st.session_state.X_test = X_test_scaled
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.scaler = fitted_scaler
    st.session_state.X_columns = X.columns.tolist()
    
    st.session_state.split_info = {
        "Tổng số mẫu": len(X),
        "Huấn luyện (Train)": len(X_train_scaled),
        "Kiểm tra (Test)": len(X_test_scaled)
    }
    st.session_state.step = 7 

def go_to_step_8_train(): st.session_state.step = 8 
def go_to_step_9_predict(): st.session_state.step = 9 
def reset_app(): st.session_state.clear(); st.rerun()

st.set_page_config(layout="wide")
st.title("Pipeline Dự đoán Doanh số")

if st.session_state.step == 0:
    st.header("Bước 1: Tải lên Dữ liệu")
    uploaded_file = st.file_uploader("Chọn file CSV", type="csv")
    if uploaded_file:
        st.session_state.df_raw = pd.read_csv(uploaded_file)
        st.session_state.step = 1
        st.rerun()

if st.session_state.step >= 1:
    with st.expander("Bước 1: Dữ liệu thô", expanded=(st.session_state.step==1)):
        st.success("Tải file thành công!")
        st.dataframe(st.session_state.df_raw.head())
        
        st.subheader("Thông tin (.info)")
        buffer = io.StringIO()
        st.session_state.df_raw.info(buf=buffer, verbose=True) 
        s = buffer.getvalue()
        lines = s.splitlines()
        s_cleaned = "\n".join(lines[1:])
        st.code(s_cleaned, language=None)

        if st.session_state.step == 1:
             st.button("Tiếp tục (Bước 2: EDA)", on_click=go_to_step_2_eda, type="primary")

if st.session_state.step >= 2:
    with st.expander("Bước 2: Khám phá Dữ liệu (EDA)", expanded=(st.session_state.step==2)):
        st.write("Hiển thị các biểu đồ cơ bản từ dữ liệu thô.")
        tab1, tab2 = st.tabs(["Biểu đồ Tương quan", "Xu hướng Thời gian"])
        with tab1:
            st.pyplot(pipeline.plot_correlation_heatmap(st.session_state.df_raw))
        with tab2:
            st.pyplot(pipeline.plot_sales_over_time(st.session_state.df_raw))
        
        if st.session_state.step == 2:
            st.button("Bắt đầu Tiền xử lý (Bước 3)", on_click=run_step_1_1, type="primary")

if st.session_state.step >= 3:
    with st.expander("Bước 3.1: Xử lý Missing Values", expanded=(st.session_state.step==3)):
        st.write("**Hành động:** Thay thế tất cả các giá trị `NaN` trong các cột `MarkDown` bằng số `0`.")
        st.code("df[markdown_cols] = df[markdown_cols].fillna(0)", language="python")
        
        st.write("**Kết quả:**")
        st.dataframe(st.session_state.df_step_1_1.head())
        if st.session_state.step == 3: st.button("Tiếp tục (Bước 3.2)", on_click=run_step_1_2)

if st.session_state.step >= 4:
    with st.expander("Bước 3.2: Xử lý Nhiễu (Âm)", expanded=(st.session_state.step==4)):
        st.write("**Hành động:** Chuyển đổi tất cả các giá trị `Weekly_Sales` âm thành `0`.")
        st.code("df.loc[df['Weekly_Sales'] < 0, 'Weekly_Sales'] = 0", language="python")
        
        st.write("**Kết quả:**")
        st.dataframe(st.session_state.df_step_1_2.head())
        if st.session_state.step == 4: st.button("Tiếp tục (Bước 3.3)", on_click=run_step_1_3)

if st.session_state.step >= 5:
    with st.expander("Bước 3.3: Feature Engineering", expanded=(st.session_state.step==5)):
        st.write("**Hành động:** Tạo đặc trưng ngày tháng và mã hóa biến Type.")
        st.code("""
# Chuyển 'Date' thành Year, Month, Week, Day
# Chuyển 'IsHoliday' (True/False) thành 1/0
# Chuyển 'Type' (A,B,C) thành 3 cột 1/0 (One-Hot)
        """, language="python")
        
        st.write("**Kết quả:**")
        st.dataframe(st.session_state.df_step_1_3.head())
        if st.session_state.step == 5: st.button("Tiếp tục (Bước 3.4)", on_click=run_step_1_4)

if st.session_state.step >= 6:
    with st.expander("Bước 3.4: Chuẩn bị Scaler", expanded=(st.session_state.step==6)):
        st.write("**Hành động:** Khởi tạo `StandardScaler` nhưng **chưa áp dụng ngay**.")
        st.code("scaler = StandardScaler() # Chỉ khởi tạo, KHÔNG fit để tránh rò rỉ dữ liệu (Data Leakage)", language="python")
        
        st.dataframe(st.session_state.df_processed.head())
        if st.session_state.step == 6:
            st.button("Chia tập & Chuẩn hóa (Bước 4)", on_click=run_step_3_split_and_scale, type="primary")

if st.session_state.step >= 7:
    if 'split_info' not in st.session_state or st.session_state.get('split_type') != 'time_series':
        df = st.session_state.df_processed.copy()
        
        if 'Date' not in df.columns:
            df['Date'] = st.session_state.df_raw['Date'].values

        target_col = 'Weekly_Sales'
        date_col = 'Date'

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(by=date_col).reset_index(drop=True)
        
        X = df.drop(columns=[target_col, date_col], errors='ignore')
        y = df[target_col]

        split_ratio = 0.8
        split_index = int(len(df) * split_ratio)

        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        cols_to_scale = [c for c in NUMERIC_FEATURES if c in X_train.columns]
        if cols_to_scale:
            scaler.fit(X_train[cols_to_scale])
            X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
            X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

        st.session_state.X_train = X_train_scaled
        st.session_state.X_test = X_test_scaled
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.X_columns = X.columns.tolist()
        
        start_train = df[date_col].iloc[0].date()
        end_train = df[date_col].iloc[split_index-1].date()
        start_test = df[date_col].iloc[split_index].date()
        end_test = df[date_col].iloc[-1].date()

        st.session_state.split_info = {
            "Loại chia": "Time-Series Split (Theo thời gian)",
            "Tỷ lệ Train/Test": f"{int(split_ratio*100)}% / {int((1-split_ratio)*100)}%",
            "Số lượng mẫu Train": len(X_train),
            "Số lượng mẫu Test": len(X_test),
            "Khoảng thời gian Train": f"{start_train} -> {end_train}",
            "Khoảng thời gian Test": f"{start_test} -> {end_test}"
        }
        st.session_state.split_type = 'time_series'

    with st.expander("Bước 4: Chia tập Train/Test & Cấu hình Mô hình", expanded=(st.session_state.step==7)):
        st.write("**1. Kết quả Chia & Chuẩn hóa (Time-Series):**")
        st.json(st.session_state.split_info)
        st.info("Dữ liệu đã được sắp xếp theo thời gian. Tập Train là quá khứ, tập Test là tương lai.")

        st.divider()
        st.write("**2. Cấu hình Tham số Huấn luyện (Hyperparameters):**")
        st.info("Tại đây bạn có thể thử nghiệm thay đổi tham số để xem mô hình thay đổi ra sao.")
        
        with st.form("train_config_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Decision Tree")
                dt_depth = st.number_input("Độ sâu tối đa (Max Depth)", 0, 50, 0)
            with col2:
                st.subheader("Random Forest")
                rf_trees = st.slider("Số lượng cây (n_estimators)", 10, 200, 50, 10)
                rf_depth = st.number_input("Độ sâu tối đa của Rừng (Max Depth)", 0, 50, 0)

            final_dt_depth = None if dt_depth == 0 else dt_depth
            final_rf_depth = None if rf_depth == 0 else rf_depth
            
            submitted_train = st.form_submit_button("Lưu Cấu hình & Bắt đầu Huấn luyện (Bước 5)", type="primary")
            
            if submitted_train:
                st.session_state.train_params = {
                    'dt_max_depth': final_dt_depth,
                    'rf_n_estimators': rf_trees,
                    'rf_max_depth': final_rf_depth
                }
                st.session_state.step = 8
                st.rerun()

if st.session_state.step >= 8:
    if st.session_state.results is None:
        params = st.session_state.get('train_params', {'rf_n_estimators': 50})
        with st.spinner(f"Đang huấn luyện với cấu hình: RF Trees={params.get('rf_n_estimators')}..."):
            
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            
            results, models = pipeline.run_training_pipeline(X_train, X_test, y_train, y_test, params=params)
            
            st.session_state.results = results
            st.session_state.models = models
            
        st.rerun()
    
    if st.session_state.results is not None:
        with st.expander("Bước 5: Kết quả & Đánh giá", expanded=(st.session_state.step==8)):
            if 'train_params' in st.session_state:
                st.caption(f"Kết quả chạy với cấu hình: {st.session_state.train_params}")
                
            results = st.session_state.results
            
            st.write("##### 1. Chỉ số Đánh giá (Metrics)")
            c1, c2, c3 = st.columns(3)
            
            res_lr = results.get('Linear Regression', {'MAE': 0, 'R-squared (R²)': 0})
            res_dt = results.get('Decision Tree', {'MAE': 0, 'R-squared (R²)': 0})
            res_rf = results.get('Random Forest', {'MAE': 0, 'R-squared (R²)': 0})

            with c1:
                st.metric("Linear Regression", f"${res_lr['MAE']:,.0f}", f"R² = {res_lr['R-squared (R²)']:.4f}")
                
            with c2:
                st.metric("Decision Tree", f"${res_dt['MAE']:,.0f}", f"R² = {res_dt['R-squared (R²)']:.4f}")
                
            with c3:
                st.metric("Random Forest", f"${res_rf['MAE']:,.0f}", f"R² = {res_rf['R-squared (R²)']:.4f}")
                
            st.divider()
            t1, t2 = st.tabs(["So sánh Hiệu suất", "Thực tế vs Dự đoán"])
            
            with t1:
                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    st.write("**So sánh MAE (Thấp hơn là tốt hơn):**")
                    df_rmse = pd.DataFrame({
                        'Model': list(results.keys()), 
                        'MAE': [v['MAE'] for v in results.values()]
                    })
                    st.bar_chart(df_rmse.set_index('Model'))
            
                with c_chart2:
                    st.write("**So sánh R² (Cao hơn là tốt hơn - Max 1.0):**")
                    df_r2 = pd.DataFrame({
                        'Model': list(results.keys()), 
                        'R²': [v['R-squared (R²)'] for v in results.values()]
                    })
                    st.bar_chart(df_r2.set_index('Model'))
                    
            with t2:
                col_chart, col_control = st.columns([2, 1])
                
                with col_control:
                    st.write("**Phân tích chi tiết:**")
                    model_names = list(st.session_state.models.keys())
                    idx_def = 2 if len(model_names) > 2 else 0
                    selected_model_name = st.selectbox("Chọn mô hình:", model_names, index=idx_def)
                    
                    if selected_model_name:
                        r2_val = results[selected_model_name]['R-squared (R²)']
                        st.write(f"**Độ chính xác:** {r2_val*100:.2f}%")
                        
                with col_chart:
                    if st.session_state.models:
                        model = st.session_state.models[selected_model_name]
                        fig = pipeline.plot_actual_vs_predicted(
                            model, 
                            st.session_state.X_test, 
                            st.session_state.y_test, 
                            selected_model_name
                        )
                        st.pyplot(fig, use_container_width=True)
                        
            st.divider()
                
            if st.session_state.step == 8:
                st.button("Tiếp tục: Dự đoán Tùy chỉnh (Bước 6)", on_click=go_to_step_9_predict, type="primary")

if st.session_state.step >= 9:
    with st.expander("Bước 6: Dự đoán Tùy chỉnh", expanded=True):
        st.write("Nhập thông tin để dự đoán doanh số:")
        
        with st.form("pred_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                store = st.number_input("Cửa hàng (Store ID)", 1, 45, 1)
                dept = st.number_input("Phòng ban (Dept ID)", 1, 99, 1)
                type_v = st.selectbox("Loại cửa hàng (Type)", ["A", "B", "C"])
                size = st.number_input("Kích thước (Size)", 30000, 250000, 150000, step=1000)
                is_holiday = st.checkbox("Là ngày lễ? (IsHoliday)", value=False)

            with col2:
                markdown1 = st.number_input("Khuyến mãi 1", 0, 10000, 0)
                markdown2 = st.number_input("Khuyến mãi 2", 0, 10000, 0)
                markdown3 = st.number_input("Khuyến mãi 3", 0, 10000, 0)
                markdown4 = st.number_input("Khuyến mãi 4", 0, 10000, 0)
                markdown5 = st.number_input("Khuyến mãi 5", 0, 10000, 0)

            with col3:
                temp = st.slider("Nhiệt độ (Temperature F)", 0, 100, 60)
                predict_date = st.date_input("Ngày dự đoán", pd.to_datetime("2012-10-26"))
                fuel = st.number_input("Giá nhiên liệu (Fuel Price)", 2.0, 5.0, 3.5, step=0.1)
                cpi = st.number_input("CPI (Chỉ số tiêu dùng)", 100.0, 250.0, 190.0)
                unemp = st.number_input("Thất nghiệp (Unemployment)", 0.0, 15.0, 8.0)
            
            submitted = st.form_submit_button("Dự đoán Doanh số", type="primary")
        
        if submitted:
            year = predict_date.year
            month = predict_date.month
            day = predict_date.day
            week = predict_date.isocalendar()[1]

            input_data = {
                'Store': store, 'Dept': dept, 'Size': size, 
                'CPI': cpi, 'Unemployment': unemp, 'Fuel_Price': fuel, 'Temperature': temp,
                'MarkDown1': markdown1, 'MarkDown2': markdown2, 'MarkDown3': markdown3, 'MarkDown4': markdown4, 'MarkDown5': markdown5,
                'IsHoliday': 1 if is_holiday else 0, 
                'Year': year, 'Month': month, 'WeekOfYear': week, 'Day': day,
                'Type_A': 1 if type_v=='A' else 0,
                'Type_B': 1 if type_v=='B' else 0,
                'Type_C': 1 if type_v=='C' else 0
            }
            
            input_df = pd.DataFrame([input_data])
            
            cols_scale = [c for c in NUMERIC_FEATURES if c in input_df.columns]
            
            if st.session_state.scaler:
                input_df[cols_scale] = st.session_state.scaler.transform(input_df[cols_scale])
            
            final_input = input_df[st.session_state.X_columns]
            
            if st.session_state.models:
                rf_model = st.session_state.models['Random Forest']
                pred = rf_model.predict(final_input)[0]
                
                st.divider()
                st.success(f"**Dự đoán Doanh số Tuần:** ${pred:,.2f}")
                
            else:
                st.error("Vui lòng huấn luyện mô hình ở Bước 5 trước.")
            
    st.button("Chạy lại", on_click=reset_app)