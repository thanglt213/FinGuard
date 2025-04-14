import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Giám sát Quỹ Tín dụng Nhân dân", layout="wide")
st.title("Phát hiện thất thoát tài sản")

# Tiền tố cho bài toán 3
prefix = "asset_loss_"

# Hàm tải dữ liệu mẫu
def load_asset_loss_sample_data():
    return pd.DataFrame({
        "Tháng": [f"Tháng {i}" for i in range(1, 13)],
        "Tài sản sổ sách": [250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305],
        "Tài sản thực tế": [250, 254, 258, 260, 255, 260, 265, 270, 275, 280, 285, 290],
        "Chi phí quản lý": [6, 7, 8, 9, 18, 15, 12, 10, 9, 8, 7, 6],
        "Giao dịch bên liên quan": [3, 4, 5, 6, 15, 12, 10, 8, 7, 6, 5, 4],
        "Tỷ lệ nợ khó đòi": [1.0, 1.2, 1.5, 1.8, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    })

# Tải dữ liệu
st.header("Tải dữ liệu")
asset_loss_uploaded_file = st.file_uploader("Chọn file CSV cho thất thoát tài sản", type="csv", key=f"{prefix}upload")
if asset_loss_uploaded_file is not None:
    asset_loss_data = pd.read_csv(asset_loss_uploaded_file)
    st.write("Dữ liệu đã tải lên:", asset_loss_data)
else:
    asset_loss_data = load_asset_loss_sample_data()
    st.write("Dữ liệu mẫu:", asset_loss_data)

# Tính toán đặc trưng
st.subheader("Cách tính toán đặc trưng")
st.write("""
- **Sai lệch tài sản (%)**: (Tài sản thực tế - Tài sản sổ sách) / Tài sản sổ sách * 100. Đo lường mức độ mất mát tài sản so với sổ sách.
- **Biến động tiền mặt (%)**: (Tài sản thực tế tháng này - Tài sản thực tế tháng trước) / Tài sản thực tế tháng trước * 100. Đo lường thay đổi tài sản thực tế qua các tháng.
""")

asset_loss_data[f"{prefix}Sai lệch tài sản"] = (asset_loss_data["Tài sản thực tế"] - asset_loss_data["Tài sản sổ sách"]) / asset_loss_data["Tài sản sổ sách"] * 100
asset_loss_data[f"{prefix}Biến động tiền mặt"] = asset_loss_data["Tài sản thực tế"].pct_change() * 100
asset_loss_data = asset_loss_data.dropna()

st.write("Dữ liệu sau khi tính toán đặc trưng:", asset_loss_data)

# Chọn đặc trưng
asset_loss_features = ["Chi phí quản lý", "Giao dịch bên liên quan", "Tỷ lệ nợ khó đòi", f"{prefix}Biến động tiền mặt"]
asset_loss_X = asset_loss_data[asset_loss_features]
asset_loss_y = asset_loss_data[f"{prefix}Sai lệch tài sản"]

# Huấn luyện mô hình
asset_loss_model = LinearRegression()
asset_loss_model.fit(asset_loss_X, asset_loss_y)
asset_loss_data[f"{prefix}Dự đoán sai lệch"] = asset_loss_model.predict(asset_loss_X)
asset_loss_mse = mean_squared_error(asset_loss_y, asset_loss_data[f"{prefix}Dự đoán sai lệch"])

# Thêm phần giải thích ngưỡng thất thoát trước khi chọn
st.subheader("Ngưỡng thất thoát")
st.write("""
- **Ngưỡng thất thoát (%)**: Đây là mức sai lệch tài sản tối đa (âm) mà bạn chấp nhận trước khi coi đó là dấu hiệu thất thoát. 
- **Cách sử dụng**: Ngưỡng này không được tính toán tự động từ dữ liệu, mà là giá trị bạn tự chọn dựa trên kinh nghiệm hoặc tiêu chuẩn của Quỹ. 
  - Nếu 'Dự đoán sai lệch' của một tháng nhỏ hơn ngưỡng (ví dụ: -6% < -5%), tháng đó sẽ được gắn cờ là có nguy cơ thất thoát.
  - Ví dụ: Với ngưỡng -5%, các tháng có sai lệch dự đoán dưới -5% sẽ được liệt kê.
- **Mục đích**: Giúp bạn lọc ra các tháng có sai lệch lớn để kiểm tra thêm.
""")

# Ngưỡng thất thoát
asset_loss_threshold = st.slider("Ngưỡng thất thoát (%)", -10.0, 0.0, -5.0, key=f"{prefix}threshold")
asset_loss_results = asset_loss_data[asset_loss_data[f"{prefix}Dự đoán sai lệch"] < asset_loss_threshold]

# Hiển thị kết quả
st.write(f"Mean Squared Error: {asset_loss_mse:.2f}")
st.write("Các tháng có nguy cơ thất thoát:", asset_loss_results[["Tháng"] + asset_loss_features + [f"{prefix}Sai lệch tài sản", f"{prefix}Dự đoán sai lệch"]])

# Trực quan hóa
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(asset_loss_data["Tháng"], asset_loss_data[f"{prefix}Sai lệch tài sản"], "b-o", label="Thực tế")
ax.plot(asset_loss_data["Tháng"], asset_loss_data[f"{prefix}Dự đoán sai lệch"], "g--o", label="Dự đoán")
ax.axhline(y=asset_loss_threshold, color="r", linestyle="--", label=f"Ngưỡng ({asset_loss_threshold}%)")
ax.legend()
ax.set_title("Phát hiện thất thoát tài sản")
ax.set_xlabel("Tháng")
ax.set_ylabel("Sai lệch tài sản (%)")
plt.xticks(rotation=45)
st.pyplot(fig)

# Tải xuống kết quả
csv = asset_loss_data.to_csv(index=False)
st.download_button(
    label="Tải xuống kết quả",
    data=csv,
    file_name=f"{prefix}result.csv",
    mime="text/csv",
    key=f"{prefix}download"
)