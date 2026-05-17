import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Financial Anomaly Detection", layout="wide")

st.title("Phát hiện giao dịch bất thường bằng Isolation Forest")
st.markdown(
    "Ứng dụng này giúp bạn upload dữ liệu kế toán, tiền xử lý và phát hiện giao dịch bất thường bằng Isolation Forest."
)

uploaded_file = st.file_uploader("Chọn file CSV của bộ dữ liệu Financial Anomaly Data", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Không đọc được file CSV: {e}")
        st.stop()

    st.subheader("1. Xem nhanh dữ liệu")
    st.write(f"Số dòng: {df.shape[0]} — Số cột: {df.shape[1]}")
    st.write(df.head(10))

    # Chọn cột số cho model
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.error("Dữ liệu không chứa cột số nào để huấn luyện Isolation Forest. Vui lòng kiểm tra lại file CSV.")
        st.stop()

    st.subheader("2. Tiền xử lý dữ liệu cơ bản")
    st.write("- Tự động loại bỏ dòng chứa giá trị thiếu ở các cột số.")
    st.write("- Chuẩn hóa dữ liệu số nếu cần để mô hình hoạt động ổn định.")

    df_numeric = df[numeric_cols].copy()
    missing_before = df_numeric.isna().sum().sum()
    df_numeric = df_numeric.dropna()
    missing_after = df_numeric.isna().sum().sum()
    st.write(f"Số giá trị thiếu trước: {missing_before}, sau khi drop: {missing_after}")
    st.write(f"Dữ liệu huấn luyện còn lại: {df_numeric.shape[0]} dòng.")

    scaler = StandardScaler()
    X = scaler.fit_transform(df_numeric)

    st.subheader("3. Huấn luyện Isolation Forest")
    contamination = st.slider(
        label="Tỷ lệ nhiễm bẩn (contamination rate)",
        min_value=0.001,
        max_value=0.2,
        value=0.01,
        step=0.001,
        format="%.3f"
    )

    n_estimators = st.number_input("Số cây (n_estimators)", min_value=50, max_value=500, value=100, step=10)
    random_state = 42

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X)
    predictions = model.predict(X)

    # Gắn nhãn và tạo bảng kết quả
    result_df = df_numeric.copy()
    result_df["anomaly_raw"] = predictions
    result_df["anomaly"] = result_df["anomaly_raw"].map({1: 0, -1: 1})
    result_df["anomaly_label"] = result_df["anomaly_raw"].map({1: "Normal", -1: "Anomaly"})
    result_df["anomaly_score"] = model.decision_function(X)

    anomaly_count = int((result_df["anomaly"] == 1).sum())
    normal_count = int((result_df["anomaly"] == 0).sum())

    st.success("Đã huấn luyện xong mô hình Isolation Forest.")
    st.write(f"Số giao dịch nhãn bất thường: **{anomaly_count}**; bình thường: **{normal_count}**.")

    st.subheader("4. Phân bố kết quả phát hiện bất thường")
    chart_data = pd.DataFrame({
        "Label": ["Normal", "Anomaly"],
        "Count": [normal_count, anomaly_count]
    })
    fig = px.bar(chart_data, x="Label", y="Count", color="Label", title="Số lượng giao dịch bình thường và bất thường")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5. Danh sách giao dịch đáng ngờ nhất")
    sorted_anomalies = result_df.sort_values(by="anomaly_score").head(20)
    st.write("Các giao dịch có điểm bất thường cao nhất (giá trị thấp hơn nghĩa là bất thường hơn):")
    st.dataframe(sorted_anomalies)

    with st.expander("Xem chi tiết các cột số đã dùng để huấn luyện"):
        st.write(numeric_cols)

    st.markdown(
        "---\n" 
        "**Ghi chú:** Nếu file CSV của bạn có thêm cột không số, app sẽ tự động chỉ dùng cột số để huấn luyện mô hình."
    )
else:
    st.info("Vui lòng upload file CSV để bắt đầu phát hiện bất thường.")
