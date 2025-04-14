import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="Giám sát Quỹ Tín dụng Nhân dân", layout="wide")
st.title("Phát hiện biến động bất thường")

# Tiền tố cho bài toán 1
prefix = "anomaly_"

# Hàm tải dữ liệu mẫu
def load_anomaly_sample_data():
    return pd.DataFrame({
        "Tháng": [f"Tháng {i}" for i in range(1, 13)],
        "Dư nợ": [120, 125, 130, 135, 200, 190, 180, 175, 170, 165, 160, 155],
        "Tiền gửi": [100, 105, 110, 115, 80, 85, 90, 95, 100, 105, 110, 115],
        "Nợ quá hạn": [2.4, 2.5, 2.6, 2.7, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        "Số giao dịch lớn": [6, 7, 8, 9, 20, 15, 12, 10, 9, 8, 7, 6]
    })

# Tải dữ liệu
st.header("Tải dữ liệu")
anomaly_uploaded_file = st.file_uploader("Chọn file CSV cho biến động bất thường", type="csv", key=f"{prefix}upload")
if anomaly_uploaded_file is not None:
    anomaly_data = pd.read_csv(anomaly_uploaded_file)
    st.write("Dữ liệu đã tải lên:", anomaly_data)
else:
    anomaly_data = load_anomaly_sample_data()
    st.write("Dữ liệu mẫu:", anomaly_data)

# Tính toán đặc trưng
st.subheader("Cách tính toán đặc trưng")
st.write("""
- **Biến động dư nợ (%)**: (Dư nợ tháng này - Dư nợ tháng trước) / Dư nợ tháng trước * 100. Đo lường mức tăng/giảm của khoản vay.
- **Biến động tiền gửi (%)**: (Tiền gửi tháng này - Tiền gửi tháng trước) / Tiền gửi tháng trước * 100. Đo lường thay đổi tiền gửi từ thành viên.
- **Tỷ lệ nợ quá hạn (%)**: Nợ quá hạn / Dư nợ * 100. Cho biết phần trăm khoản vay không được trả đúng hạn.
- **Tỷ lệ sử dụng vốn huy động (%)**: Dư nợ / Tiền gửi * 100. Đo lường mức độ Quỹ dùng tiền gửi để cho vay.
""")

anomaly_data[f"{prefix}Biến động dư nợ"] = anomaly_data["Dư nợ"].pct_change() * 100
anomaly_data[f"{prefix}Biến động tiền gửi"] = anomaly_data["Tiền gửi"].pct_change() * 100
anomaly_data[f"{prefix}Tỷ lệ nợ quá hạn"] = anomaly_data["Nợ quá hạn"] / anomaly_data["Dư nợ"] * 100
anomaly_data[f"{prefix}Tỷ lệ sử dụng vốn huy động"] = anomaly_data["Dư nợ"] / anomaly_data["Tiền gửi"] * 100
anomaly_data = anomaly_data.dropna()

st.write("Dữ liệu sau khi tính toán đặc trưng:", anomaly_data)

# Chọn đặc trưng
anomaly_features = [f"{prefix}Biến động dư nợ", f"{prefix}Biến động tiền gửi", f"{prefix}Tỷ lệ nợ quá hạn", f"{prefix}Tỷ lệ sử dụng vốn huy động"]
anomaly_X = anomaly_data[anomaly_features]

# Huấn luyện mô hình
anomaly_contamination = st.slider("Tỷ lệ bất thường (contamination)", 0.05, 0.5, 0.2, key=f"{prefix}contamination")
anomaly_model = IsolationForest(contamination=anomaly_contamination, random_state=42)
anomaly_model.fit(anomaly_X)
anomaly_data[f"{prefix}Anomaly"] = anomaly_model.predict(anomaly_X)

# Hiển thị kết quả
anomaly_results = anomaly_data[anomaly_data[f"{prefix}Anomaly"] == -1]
st.write("Các tháng bất thường:", anomaly_results) #[["Tháng"] + anomaly_features])

# Trực quan hóa
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(anomaly_data.index, anomaly_data[f"{prefix}Biến động dư nợ"], c=anomaly_data[f"{prefix}Anomaly"], cmap="coolwarm", s=100)
ax.set_xticks(anomaly_data.index)
ax.set_xticklabels(anomaly_data["Tháng"], rotation=45)
ax.set_xlabel("Tháng")
ax.set_ylabel("Biến động dư nợ (%)")
ax.set_title("Phát hiện biến động bất thường")
st.pyplot(fig)

# Tải xuống kết quả
csv = anomaly_data.to_csv(index=False)
st.download_button(
    label="Tải xuống kết quả",
    data=csv,
    file_name=f"{prefix}result.csv",
    mime="text/csv",
    key=f"{prefix}download"
)