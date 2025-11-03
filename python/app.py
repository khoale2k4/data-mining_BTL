import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import io
import pipeline 
from pipeline import NUMERIC_FEATURES 

if 'step' not in st.session_state:
    st.session_state.step = 0 .
    st.session_state.df_raw = None
    st.session_state.df_step_1_1 = None
    st.session_state.df_step_1_2 = None
    st.session_state.df_step_1_3 = None
    st.session_state.df_processed = None 
    st.session_state.split_info = None
    st.session_state.results = None
    st.session_state.models = None  
    st.session_state.scaler = None 
    st.session_state.X_columns = None 

def go_to_step_2_eda():
    st.session_state.step = 2

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
    df_processed, scaler = pipeline.step_1_4_scale_data(st.session_state.df_step_1_3)
    st.session_state.df_processed = df_processed
    st.session_state.scaler = scaler 
    st.session_state.step = 6

def run_step_3_split():
    y = st.session_state.df_processed['Weekly_Sales']
    X = st.session_state.df_processed.drop('Weekly_Sales', axis=1)
    st.session_state.X_columns = X.columns.tolist() 
    st.session_state.split_info = {
        "Tổng số mẫu": len(X),
        "Số mẫu huấn luyện (train)": int(len(X) * 0.8),
        "Số mẫu kiểm tra (test)": int(len(X) * 0.2)
    }
    st.session_state.step = 7 

def go_to_step_8_train():
    st.session_state.step = 8 

def go_to_step_9_predict():
    st.session_state.step = 9 

def reset_app():
    st.session_state.clear() 
    st.rerun() 

st.set_page_config(layout="wide")
st.title("Trình diễn Pipeline Học máy: Dự đoán Doanh số")

st.header("Bước 1: Tải lên Dữ liệu")
uploaded_file = st.file_uploader("Chọn file CSV của bạn", type="csv")

if uploaded_file is None and st.session_state.step == 0:
    st.info("Chờ bạn tải lên file CSV để bắt đầu...")

if uploaded_file is not None and st.session_state.step == 0:
    st.session_state.df_raw = pd.read_csv(uploaded_file)
    st.session_state.step = 1 
    st.rerun()

if st.session_state.step >= 1:
    with st.expander("Bước 1: Tải lên Dữ liệu", expanded=(st.session_state.step == 1)):
        st.success("Tải file thành công!")
        st.dataframe(st.session_state.df_raw.head())
        
        st.subheader("Thông tin Dữ liệu Thô (.info())")
        buffer = io.StringIO()
        st.session_state.df_raw.info(buf=buffer, verbose=True) 
        s = buffer.getvalue()
        lines = s.splitlines()
        s_cleaned = "\n".join(lines[1:])
        st.code(s_cleaned, language=None)
        
        if st.session_state.step == 1:
             st.button("Tiếp tục (Bước 2: Khám phá Dữ liệu)", on_click=go_to_step_2_eda, type="primary")

if st.session_state.step >= 2:
    with st.expander("Bước 2: Khám phá Dữ liệu (EDA)", expanded=(st.session_state.step == 2)):
        st.write("Hiển thị các biểu đồ cơ bản từ dữ liệu thô để hiểu rõ hơn.")
        
        st.subheader("Biểu đồ Tương quan (Heatmap)")
        with st.spinner("Đang vẽ biểu đồ heatmap..."):
            fig_heatmap = pipeline.plot_correlation_heatmap(st.session_state.df_raw)
            if fig_heatmap:
                st.pyplot(fig_heatmap, use_container_width=True) 
            else:
                st.error("Lỗi khi vẽ heatmap.")
        
        if st.session_state.step == 2:
            st.button("Bắt đầu Tiền xử lý (Bước 3)", on_click=run_step_1_1, type="primary")

if st.session_state.step >= 3:
    with st.expander("Bước 3.1: Xử lý Giá trị thiếu (MarkDowns)", expanded=(st.session_state.step == 3)):
        st.write("**Hành động:** Thay thế tất cả các giá trị `NaN` trong các cột `MarkDown` bằng số `0`.")
        st.code("df[markdown_cols] = df[markdown_cols].fillna(0)", language="python")
        st.write("**Kết quả:**")
        st.dataframe(st.session_state.df_step_1_1.head())
        if st.session_state.step == 3:
            st.button("Tiếp tục (Bước 3.2)", on_click=run_step_1_2)

if st.session_state.step >= 4:
    with st.expander("Bước 3.2: Xử lý Nhiễu (Weekly_Sales âm)", expanded=(st.session_state.step == 4)):
        st.write("**Hành động:** Chuyển đổi tất cả các giá trị `Weekly_Sales` âm (nếu có) thành `0`.")
        st.code("df.loc[df['Weekly_Sales'] < 0, 'Weekly_Sales'] = 0", language="python")
        st.write("**Kết quả:**")
        st.dataframe(st.session_state.df_step_1_2.head())
        if st.session_state.step == 4:
            st.button("Tiếp tục (Bước 3.3)", on_click=run_step_1_3)

if st.session_state.step >= 5:
    with st.expander("Bước 3.3: Tạo đặc trưng (Feature Engineering)", expanded=(st.session_state.step == 5)):
        st.write("**Hành động:** Chuyển đổi các cột `Date`, `IsHoliday`, và `Type` thành định dạng số.")
        st.code("# Chuyển 'Date' thành Year, Month, Week, Day\n# Chuyển 'IsHoliday' (True/False) thành 1/0\n# Chuyển 'Type' (A,B,C) thành 3 cột 1/0 (One-Hot)", language="python")
        st.write("**Kết quả:**")
        st.dataframe(st.session_state.df_step_1_3.head())
        if st.session_state.step == 5:
            st.button("Tiếp tục (Bước 3.4)", on_click=run_step_1_4)

if st.session_state.step >= 6:
    with st.expander("Bước 3.4: Chuẩn hóa Dữ liệu (StandardScaler)", expanded=(st.session_state.step == 6)):
        st.write("**Hành động:** Áp dụng `StandardScaler` (Z-score) cho tất cả các cột số liên tục.")
        st.code("df[numeric_features] = scaler.fit_transform(df[numeric_features])", language="python")
        st.write("**Kết quả:**")
        st.dataframe(st.session_state.df_processed.head())
        st.success("Tiền xử lý hoàn tất!")
        if st.session_state.step == 6:
            st.button("Tiếp tục (Bước 4: Chia Dữ liệu)", on_click=run_step_3_split, type="primary")

if st.session_state.step >= 7:
    with st.expander("Bước 4: Chia tập Huấn luyện / Kiểm tra", expanded=(st.session_state.step == 7)):
        st.write("Chia dữ liệu đã xử lý thành 2 phần (80% Huấn luyện, 20% Kiểm tra).")
        st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)", language="python")
        st.write("**Kết quả:**")
        st.json(st.session_state.split_info)
        if st.session_state.step == 7:
            st.button("Bắt đầu Huấn luyện (Bước 5)", on_click=go_to_step_8_train, type="primary")

if st.session_state.step >= 8:
    if st.session_state.results is None:
        with st.spinner("Đang huấn luyện tất cả 3 mô hình (Random Forest có thể mất vài phút)..."):
            y = st.session_state.df_processed['Weekly_Sales']
            X = st.session_state.df_processed.drop('Weekly_Sales', axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            results, models = pipeline.run_training_pipeline(X_train, X_test, y_train, y_test)
            st.session_state.results = results
            st.session_state.models = models 
        st.rerun() 

    with st.expander("Bước 5: Huấn luyện 3 Mô hình", expanded=False):
        st.write("Đã huấn luyện 3 mô hình trên tập Huấn luyện (80% dữ liệu).")
        st.success("Huấn luyện hoàn tất!")

    with st.expander("Bước 6: Kết quả Đánh giá & So sánh", expanded=(st.session_state.step == 8)):
        st.success("Đã hoàn tất toàn bộ pipeline!")
        
        results = st.session_state.results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Linear Regression")
            st.metric("R-squared (R²)", f"{results['Linear Regression']['R-squared (R²)']:.4f}")
            st.metric("RMSE", f"${results['Linear Regression']['RMSE']:,.2f}")
            st.metric("MAE", f"${results['Linear Regression']['MAE']:,.2f}")
            st.metric("Thời gian Huấn luyện", f"{results['Linear Regression']['Time']:.2f} giây")

        with col2:
            st.subheader("Decision Tree")
            st.metric("R-squared (R²)", f"{results['Decision Tree']['R-squared (R²)']:.4f}")
            st.metric("RMSE", f"${results['Decision Tree']['RMSE']:,.2f}")
            st.metric("MAE", f"${results['Decision Tree']['MAE']:,.2f}")
            st.metric("Thời gian Huấn luyện", f"{results['Decision Tree']['Time']:.2f} giây")

        with col3:
            st.subheader("Random Forest")
            st.metric("R-squared (R²)", f"{results['Random Forest']['R-squared (R²)']:.4f}")
            st.metric("RMSE", f"${results['Random Forest']['RMSE']:,.2f}")
            st.metric("MAE", f"${results['Random Forest']['MAE']:,.2f}")
            st.metric("Thời gian Huấn luyện", f"{results['Random Forest']['Time']:.2f} giây")

        st.divider()
        st.subheader("Trực quan hóa So sánh Hiệu suất")
        chart_data = {'Model': list(results.keys()), 'RMSE': [v['RMSE'] for v in results.values()], 'R²': [v['R-squared (R²)'] for v in results.values()]}
        df_chart = pd.DataFrame(chart_data)
        st.write("**So sánh RMSE (Lỗi - Càng thấp càng tốt)**")
        st.bar_chart(df_chart.set_index('Model')['RMSE'])
        st.write("**So sánh R² (Độ phù hợp - Càng cao càng tốt)**")
        st.bar_chart(df_chart.set_index('Model')['R²'])
        
        if st.session_state.step == 8:
            st.button("Tiếp tục (Bước 7: Dự đoán Tùy chỉnh)", on_click=go_to_step_9_predict, type="primary")

if st.session_state.step >= 9:
    with st.expander("Bước 7: Dự đoán Tùy chỉnh", expanded=True):
        st.write("Chọn một mô hình và nhập dữ liệu đầu vào để nhận dự đoán doanh số.")
        
        models = st.session_state.models
        scaler = st.session_state.scaler
        X_columns = st.session_state.X_columns
        
        model_name = st.selectbox("Chọn mô hình để dự đoán:", list(models.keys()))
        model_to_use = models[model_name]
        
        st.subheader("Nhập thông tin cho dự đoán:")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Thông tin Cơ bản")
                store = st.number_input("Store", 1, 45, 1)
                dept = st.number_input("Dept", 1, 99, 1)
                is_holiday = st.selectbox("IsHoliday?", (0, 1))
                type_val = st.selectbox("Type", ("A", "B", "C"))
            
            with col2:
                st.write("Thông tin Kinh tế & Kích thước")
                temperature = st.number_input("Temperature (F)", -10.0, 110.0, 50.0, format="%.2f")
                fuel_price = st.number_input("Fuel_Price", 2.0, 5.0, 3.5, format="%.3f")
                cpi = st.number_input("CPI", 100.0, 250.0, 170.0, format="%.2f")
                unemployment = st.number_input("Unemployment", 3.0, 15.0, 8.0, format="%.2f")
                size = st.number_input("Size (sq. ft.)", 30000, 250000, 150000)
            
            with col3:
                st.write("Thông tin Ngày tháng")
                year = st.number_input("Year", 2010, 2025, 2011)
                month = st.number_input("Month", 1, 12, 11)
                week = st.number_input("WeekOfYear", 1, 52, 44)
                day = st.number_input("Day", 1, 31, 2)

            st.write("Thông tin MarkDown (nếu có)")
            col4, col5, col6 = st.columns(3)
            with col4:
                md1 = st.number_input("MarkDown1", 0.0, format="%.2f")
                md2 = st.number_input("MarkDown2", 0.0, format="%.2f")
            with col5:
                md3 = st.number_input("MarkDown3", 0.0, format="%.2f")
                md4 = st.number_input("MarkDown4", 0.0, format="%.2f")
            with col6:
                md5 = st.number_input("MarkDown5", 0.0, format="%.2f")
                
            submitted = st.form_submit_button("Dự đoán")
            
        if submitted:
            with st.spinner("Đang xử lý đầu vào và dự đoán..."):
                input_data = {
                    'Store': store, 'Dept': dept, 'IsHoliday': is_holiday,
                    'Temperature': temperature, 'Fuel_Price': fuel_price,
                    'MarkDown1': md1, 'MarkDown2': md2, 'MarkDown3': md3,
                    'MarkDown4': md4, 'MarkDown5': md5,
                    'CPI': cpi, 'Unemployment': unemployment, 'Size': size,
                    'Year': year, 'Month': month, 'WeekOfYear': week, 'Day': day,
                    'Type_A': 1 if type_val == 'A' else 0,
                    'Type_B': 1 if type_val == 'B' else 0,
                    'Type_C': 1 if type_val == 'C' else 0
                }
                
                input_df = pd.DataFrame([input_data])
                
                features_to_scale = [col for col in NUMERIC_FEATURES if col in input_df.columns]
                input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])
                
                try:
                    input_df = input_df[X_columns]
                    prediction = model_to_use.predict(input_df)
                    
                    st.success(f"**Dự đoán Doanh số (Weekly_Sales) từ mô hình {model_name}:**")
                    st.subheader(f"${prediction[0]:,.2f}")
                except Exception as e:
                    st.error(f"Lỗi sắp xếp cột hoặc dự đoán. Đảm bảo file CSV của bạn có đủ các cột. Lỗi: {e}")
                
        st.button("Chạy lại từ đầu", on_click=reset_app)