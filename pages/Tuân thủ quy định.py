import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Giám sát Quỹ Tín dụng Nhân dân", layout="wide")
st.title("Kiểm tra tuân thủ an toàn vốn và nợ xấu")

# Tiền tố cho bài toán
prefix = "compliance_"

# Hàm tải dữ liệu mẫu
def load_compliance_sample_data():
    return pd.DataFrame({
        "Quỹ": [f"Quỹ {i}" for i in range(1, 21)],
        "Vốn chủ sở hữu": [25, 20, 30, 10, 22, 35, 12, 28, 20, 26, 8, 23, 29, 9, 24, 32, 19, 27, 15, 31],
        "Tổng tài sản": [200, 180, 220, 150, 190, 250, 160, 210, 170, 200, 140, 185, 230, 145, 195, 260, 155, 205, 175, 240],
        "Tài sản có rủi ro": [150, 140, 160, 140, 145, 180, 160, 165, 135, 155, 130, 140, 170, 135, 150, 190, 125, 160, 150, 175],
        "Nợ xấu": [2.0, 3.0, 1.5, 8.0, 2.5, 1.0, 6.0, 1.8, 3.5, 2.2, 9.0, 2.8, 1.7, 7.0, 3.0, 1.2, 4.5, 2.0, 5.0, 1.8],
        "Tổng dư nợ": [120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215]
    })

# Tải dữ liệu
st.header("Tải dữ liệu")
compliance_uploaded_file = st.file_uploader("Chọn file CSV cho kiểm tra tuân thủ", type="csv", key=f"{prefix}upload")
if compliance_uploaded_file is not None:
    compliance_data = pd.read_csv(compliance_uploaded_file)
    st.write("Dữ liệu đã tải lên:", compliance_data)
else:
    compliance_data = load_compliance_sample_data()
    st.write("Dữ liệu mẫu:", compliance_data)

# Tính toán đặc trưng
st.subheader("Cách tính toán đặc trưng")
st.write("""
- **Tỷ lệ an toàn vốn (CAR - %)**: Vốn chủ sở hữu / Tài sản có rủi ro * 100. Đo lường khả năng Quỹ chịu đựng rủi ro tài chính.
- **Tỷ lệ nợ xấu (%)**: Nợ xấu / Tổng dư nợ * 100. Đo lường mức độ rủi ro từ các khoản vay không thu hồi được.
""")

compliance_data[f"{prefix}CAR"] = compliance_data["Vốn chủ sở hữu"] / compliance_data["Tài sản có rủi ro"] * 100
compliance_data[f"{prefix}Tỷ lệ nợ xấu"] = compliance_data["Nợ xấu"] / compliance_data["Tổng dư nợ"] * 100

st.write("Dữ liệu sau khi tính toán đặc trưng:", compliance_data)

# Định nghĩa ngưỡng tuân thủ
st.subheader("Ngưỡng tuân thủ")
st.write("""
- **Tỷ lệ an toàn vốn tối thiểu (CAR)**: Theo quy định (ví dụ: Ngân hàng Nhà nước Việt Nam), thường là 9%. Quỹ có CAR < 9% vi phạm an toàn vốn.
- **Tỷ lệ nợ xấu tối đa**: Thường là 3%. Quỹ có tỷ lệ nợ xấu > 3% vi phạm giới hạn nợ xấu.
""")

car_threshold = st.slider("Ngưỡng tỷ lệ an toàn vốn tối thiểu (%)", 5.0, 15.0, 8.0, key=f"{prefix}car_threshold")
bad_debt_threshold = st.slider("Ngưỡng tỷ lệ nợ xấu tối đa (%)", 1.0, 10.0, 3.0, key=f"{prefix}bad_debt_threshold")

# Kiểm tra tuân thủ
compliance_data[f"{prefix}CAR_Compliance"] = compliance_data[f"{prefix}CAR"] >= car_threshold
compliance_data[f"{prefix}Bad_Debt_Compliance"] = compliance_data[f"{prefix}Tỷ lệ nợ xấu"] <= bad_debt_threshold
compliance_data[f"{prefix}Overall_Compliance"] = compliance_data[f"{prefix}CAR_Compliance"] & compliance_data[f"{prefix}Bad_Debt_Compliance"]

# Hiển thị kết quả
st.subheader("Kết quả kiểm tra tuân thủ")
st.write("Dữ liệu sau khi kiểm tra tuân thủ:", compliance_data[["Quỹ", f"{prefix}CAR", f"{prefix}Tỷ lệ nợ xấu", f"{prefix}CAR_Compliance", f"{prefix}Bad_Debt_Compliance", f"{prefix}Overall_Compliance"]])

# Các Quỹ vi phạm
non_compliant = compliance_data[compliance_data[f"{prefix}Overall_Compliance"] == False]
st.write("Các Quỹ không tuân thủ:", non_compliant[["Quỹ", f"{prefix}CAR", f"{prefix}Tỷ lệ nợ xấu"]])

# Trực quan hóa
st.subheader("Biểu đồ kiểm tra tuân thủ")

# Biểu đồ CAR
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    x="Quỹ", 
    y=f"{prefix}CAR", 
    hue=f"{prefix}CAR_Compliance", 
    data=compliance_data, 
    ax=ax1, 
    palette="coolwarm", 
    s=100,  # Tăng kích thước điểm
    alpha=1.0  # Độ đậm tối đa (không trong suốt)
)
ax1.axhline(y=car_threshold, color="r", linestyle="--", label=f"Ngưỡng CAR ({car_threshold}%)")
ax1.set_title("Tỷ lệ an toàn vốn (CAR)")
ax1.set_ylabel("CAR (%)")
ax1.set_xticklabels(compliance_data["Quỹ"], rotation=45)
ax1.legend()
st.pyplot(fig1)

# Biểu đồ tỷ lệ nợ xấu
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    x="Quỹ", 
    y=f"{prefix}Tỷ lệ nợ xấu", 
    hue=f"{prefix}Bad_Debt_Compliance", 
    data=compliance_data, 
    ax=ax2, 
    palette="coolwarm", 
    s=100,  # Tăng kích thước điểm
    alpha=1.0  # Độ đậm tối đa (không trong suốt)
)
ax2.axhline(y=bad_debt_threshold, color="r", linestyle="--", label=f"Ngưỡng nợ xấu ({bad_debt_threshold}%)")
ax2.set_title("Tỷ lệ nợ xấu")
ax2.set_ylabel("Tỷ lệ nợ xấu (%)")
ax2.set_xticklabels(compliance_data["Quỹ"], rotation=45)
ax2.legend()
st.pyplot(fig2)

# Tải xuống kết quả
csv = compliance_data.to_csv(index=False)
st.download_button(
    label="Tải xuống kết quả",
    data=csv,
    file_name=f"{prefix}result.csv",
    mime="text/csv",
    key=f"{prefix}download"
)