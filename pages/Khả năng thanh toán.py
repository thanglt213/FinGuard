import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Giám sát Quỹ Tín dụng Nhân dân", layout="wide")
st.title("Phát hiện mất khả năng thanh toán")

# Tiền tố cho bài toán 2
prefix = "insolvency_"

# Hàm tải dữ liệu mẫu
def load_insolvency_sample_data():
    return pd.DataFrame({
        "Quỹ": [f"Quỹ {i}" for i in range(1, 21)],
        "Tiền mặt": [12, 8, 15, 3, 10, 20, 5, 18, 7, 14, 2, 9, 16, 4, 11, 19, 6, 13, 8, 17],
        "Nợ ngắn hạn": [50, 60, 45, 80, 55, 40, 70, 50, 65, 48, 90, 52, 47, 75, 58, 42, 68, 53, 62, 46],
        "Dòng tiền ròng": [3, -2, 1, -6, 0, 4, -4, 2, -3, 1, -8, 0, 3, -5, -1, 5, -3, 2, -2, 4],
        "Vốn chủ sở hữu": [25, 20, 30, 15, 22, 35, 18, 28, 20, 26, 12, 23, 29, 17, 24, 32, 19, 27, 21, 31],
        "Nợ quá hạn": [2.0, 3.0, 1.5, 8.0, 2.5, 1.0, 4.0, 1.8, 3.5, 2.2, 9.0, 2.8, 1.7, 6.0, 3.0, 1.2, 4.5, 2.0, 3.2, 1.8]
    })

# Tải dữ liệu
st.header("Tải dữ liệu")
insolvency_uploaded_file = st.file_uploader("Chọn file CSV cho mất thanh khoản", type="csv", key=f"{prefix}upload")
if insolvency_uploaded_file is not None:
    insolvency_data = pd.read_csv(insolvency_uploaded_file)
    st.write("Dữ liệu đã tải lên:", insolvency_data)
else:
    insolvency_data = load_insolvency_sample_data()
    st.write("Dữ liệu mẫu:", insolvency_data)

# Tính toán đặc trưng
st.subheader("Cách tính toán đặc trưng")
st.write("""
- **Tỷ lệ thanh khoản (%)**: Tiền mặt / Nợ ngắn hạn * 100. Đo lường khả năng Quỹ trả nợ ngắn hạn bằng tiền mặt.
- **Tỷ lệ nợ/vốn (%)**: Nợ ngắn hạn / Vốn chủ sở hữu * 100. Cho biết mức độ Quỹ phụ thuộc vào nợ so với vốn tự có.
- **Tỷ lệ tài sản thanh khoản (%)**: Tiền mặt / (Tiền mặt + Nợ ngắn hạn) * 100. Đo lường tỷ lệ tiền mặt trong tổng tài sản ngắn hạn.
- **Tỷ lệ nợ quá hạn (%)**: Nợ quá hạn / Nợ ngắn hạn * 100. Cho biết phần trăm nợ ngắn hạn không được trả đúng hạn.
""")

insolvency_data[f"{prefix}Tỷ lệ thanh khoản"] = insolvency_data["Tiền mặt"] / insolvency_data["Nợ ngắn hạn"] * 100
insolvency_data[f"{prefix}Tỷ lệ nợ/vốn"] = insolvency_data["Nợ ngắn hạn"] / insolvency_data["Vốn chủ sở hữu"] * 100
insolvency_data[f"{prefix}Tỷ lệ tài sản thanh khoản"] = insolvency_data["Tiền mặt"] / (insolvency_data["Tiền mặt"] + insolvency_data["Nợ ngắn hạn"]) * 100
insolvency_data[f"{prefix}Tỷ lệ nợ quá hạn"] = insolvency_data["Nợ quá hạn"] / insolvency_data["Nợ ngắn hạn"] * 100

st.write("Dữ liệu sau khi tính toán đặc trưng:", insolvency_data)

# Chọn đặc trưng
insolvency_features = [f"{prefix}Tỷ lệ thanh khoản", f"{prefix}Tỷ lệ nợ/vốn", f"{prefix}Tỷ lệ tài sản thanh khoản", "Dòng tiền ròng", f"{prefix}Tỷ lệ nợ quá hạn"]
insolvency_X = insolvency_data[insolvency_features]

# Chuẩn hóa dữ liệu
insolvency_scaler = StandardScaler()
insolvency_X_scaled = insolvency_scaler.fit_transform(insolvency_X)

# Huấn luyện mô hình
insolvency_contamination = st.slider("Tỷ lệ bất thường (contamination)", 0.05, 0.5, 0.2, key=f"{prefix}contamination")
insolvency_model = IsolationForest(contamination=insolvency_contamination, random_state=42)
insolvency_model.fit(insolvency_X_scaled)
insolvency_data[f"{prefix}Anomaly_Score"] = insolvency_model.decision_function(insolvency_X_scaled)

# In dữ liệu sau khi huấn luyện mô hình
st.subheader("Dữ liệu sau khi huấn luyện mô hình")
st.write("Dữ liệu bao gồm điểm bất thường (Anomaly Score) cho từng Quỹ:", insolvency_data[["Quỹ"] + insolvency_features + [f"{prefix}Anomaly_Score"]])

# Giải thích điểm ngưỡng bất thường trước khi chọn
st.subheader("Điểm ngưỡng bất thường")
st.write("""
- **Công thức tính điểm bất thường (Anomaly Score)**: 
  - Điểm này được Isolation Forest tính toán dựa trên mức độ 'khác thường' của mỗi Quỹ so với toàn bộ dữ liệu.
  - Nó không có công thức đơn giản như phép tính, mà dựa trên thuật toán cây quyết định: 
    - Mỗi Quỹ được đưa qua nhiều cây quyết định ngẫu nhiên.
    - Độ dài đường đi trung bình (path length) của Quỹ trong các cây càng ngắn (so với trung bình), điểm càng thấp (âm hơn), nghĩa là Quỹ càng bất thường.
  - Kết quả: Điểm gần 0 hoặc dương = bình thường; điểm âm lớn (ví dụ: -0.3) = bất thường.

- **Điểm ngưỡng bất thường (Threshold Score)**:
  - Đây là giá trị bạn chọn để phân loại Quỹ nào là 'rủi ro'.
  - **Cách sử dụng**: Nếu Anomaly Score của Quỹ < Threshold Score, Quỹ được gắn cờ 'Risk = 1' (nguy cơ mất thanh khoản).
  - Ví dụ: Với ngưỡng -0.1, Quỹ có điểm -0.25 sẽ bị gắn cờ, còn Quỹ có điểm 0.05 thì không.
  - **Mục đích**: Giúp bạn kiểm soát độ nhạy của phát hiện. Ngưỡng cao (gần 0) sẽ gắn cờ nhiều Quỹ hơn; ngưỡng thấp (như -0.5) chỉ gắn cờ các Quỹ rất bất thường.
""")

# Chọn ngưỡng bất thường
insolvency_threshold_score = st.slider("Chọn ngưỡng điểm bất thường", -0.5, 0.0, -0.1, key=f"{prefix}threshold")
insolvency_data[f"{prefix}Risk"] = (insolvency_data[f"{prefix}Anomaly_Score"] < insolvency_threshold_score).astype(int)

# Hiển thị kết quả
insolvency_results = insolvency_data[insolvency_data[f"{prefix}Risk"] == 1]
st.write("Các Quỹ có nguy cơ mất thanh khoản:", insolvency_results[["Quỹ"] + insolvency_features + [f"{prefix}Anomaly_Score"]])

# Trực quan hóa
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x=f"{prefix}Tỷ lệ thanh khoản",
    y="Dòng tiền ròng",
    hue=f"{prefix}Risk",
    size=f"{prefix}Anomaly_Score",
    sizes=(50, 200),
    data=insolvency_data,
    palette="coolwarm",
    ax=ax
)
for i, row in insolvency_data.iterrows():
    ax.text(row[f"{prefix}Tỷ lệ thanh khoản"] + 0.5, row["Dòng tiền ròng"], row["Quỹ"], fontsize=9)
ax.set_title("Phát hiện nguy cơ mất thanh khoản")
st.pyplot(fig)

# Tải xuống kết quả
csv = insolvency_data.to_csv(index=False)
st.download_button(
    label="Tải xuống kết quả",
    data=csv,
    file_name=f"{prefix}result.csv",
    mime="text/csv",
    key=f"{prefix}download"
)