import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Đánh giá mức độ rủi ro tín dụng")

# Tiền tố cho bài toán
prefix = "credit_risk_"

# Hàm tải dữ liệu mẫu (điều chỉnh để phân bố đều hơn)
def load_credit_risk_sample_data():
    return pd.DataFrame({
        "Quỹ": [f"Quỹ {i}" for i in range(1, 21)],
        "Tổng dư nợ": [120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215],
        "Nợ xấu": [2.0, 3.0, 1.5, 8.0, 2.5, 1.0, 6.0, 1.8, 3.5, 2.2, 9.0, 2.8, 1.7, 7.0, 3.0, 1.2, 4.5, 2.0, 5.0, 1.8],
        "Tổng tiền gửi": [130, 120, 140, 110, 145, 150, 115, 160, 150, 170, 120, 155, 190, 130, 165, 200, 140, 180, 150, 220],
        # Điều chỉnh Risk_Label để đảm bảo có đủ 3 lớp
        "Risk_Label": [0, 1, 0, 2, 0, 0, 2, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 1]
    })

# Tải dữ liệu
st.header("Tải dữ liệu")
credit_risk_uploaded_file = st.file_uploader("Chọn file CSV cho đánh giá rủi ro tín dụng", type="csv", key=f"{prefix}upload")
if credit_risk_uploaded_file is not None:
    credit_risk_data = pd.read_csv(credit_risk_uploaded_file)
    st.write("Dữ liệu đã tải lên:", credit_risk_data)
else:
    credit_risk_data = load_credit_risk_sample_data()
    st.write("Dữ liệu mẫu:", credit_risk_data)

# Tính toán đặc trưng
st.subheader("Cách tính toán đặc trưng")
st.write("""
- **Tỷ lệ nợ xấu (%)**: Nợ xấu / Tổng dư nợ * 100. Đo lường mức độ rủi ro từ các khoản vay không thu hồi được.
- **Tỷ lệ sử dụng vốn (%)**: Tổng dư nợ / Tổng tiền gửi * 100. Đo lường mức độ sử dụng vốn huy động để cho vay.
""")

credit_risk_data[f"{prefix}Tỷ lệ nợ xấu"] = credit_risk_data["Nợ xấu"] / credit_risk_data["Tổng dư nợ"] * 100
credit_risk_data[f"{prefix}Tỷ lệ sử dụng vốn"] = credit_risk_data["Tổng dư nợ"] / credit_risk_data["Tổng tiền gửi"] * 100

st.write("Dữ liệu sau khi tính toán đặc trưng:", credit_risk_data)

# Chọn đặc trưng để huấn luyện
features = [f"{prefix}Tỷ lệ nợ xấu", f"{prefix}Tỷ lệ sử dụng vốn"]
X = credit_risk_data[features]
y = credit_risk_data["Risk_Label"]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu để huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình
model = LogisticRegression(multi_class="multinomial", max_iter=1000)
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_scaled)
credit_risk_data[f"{prefix}Risk_Prediction"] = y_pred
credit_risk_data[f"{prefix}Risk_Level"] = credit_risk_data[f"{prefix}Risk_Prediction"].map({0: "Thấp", 1: "Trung bình", 2: "Cao"})

# Hiển thị kết quả
st.subheader("Kết quả đánh giá rủi ro tín dụng")
st.write("Dữ liệu sau khi dự đoán:", credit_risk_data[["Quỹ"] + features + [f"{prefix}Risk_Level"]])

# Phân loại theo mức rủi ro
st.subheader("Phân loại Quỹ theo mức rủi ro")
low_risk = credit_risk_data[credit_risk_data[f"{prefix}Risk_Prediction"] == 0]
medium_risk = credit_risk_data[credit_risk_data[f"{prefix}Risk_Prediction"] == 1]
high_risk = credit_risk_data[credit_risk_data[f"{prefix}Risk_Prediction"] == 2]

st.write("1. Quỹ có rủi ro thấp:", low_risk[["Quỹ"] + features + [f"{prefix}Risk_Level"]])
st.write("2. Quỹ có rủi ro trung bình:", medium_risk[["Quỹ"] + features + [f"{prefix}Risk_Level"]])
st.write("3. Quỹ có rủi ro cao:", high_risk[["Quỹ"] + features + [f"{prefix}Risk_Level"]])

# Báo cáo đánh giá mô hình
st.subheader("Đánh giá mô hình")
try:
    report = classification_report(y_test, model.predict(X_test), labels=[0, 1, 2], target_names=["Thấp", "Trung bình", "Cao"])
    st.text("Báo cáo phân loại:\n" + report)
except ValueError as e:
    st.warning(f"Cảnh báo: Không đủ dữ liệu cho tất cả các lớp trong tập kiểm tra. Lỗi: {e}")
    unique_classes = np.unique(y_test)
    st.text(f"Các lớp có trong tập kiểm tra: {unique_classes}")
    if len(unique_classes) > 0:
        report = classification_report(y_test, model.predict(X_test), labels=unique_classes, target_names=[{0: "Thấp", 1: "Trung bình", 2: "Cao"}[i] for i in unique_classes])
        st.text("Báo cáo phân loại (chỉ với các lớp có dữ liệu):\n" + report)

# Trực quan hóa
st.subheader("Biểu đồ đánh giá rủi ro tín dụng")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x=f"{prefix}Tỷ lệ nợ xấu",
    y=f"{prefix}Tỷ lệ sử dụng vốn",
    hue=f"{prefix}Risk_Level",
    data=credit_risk_data,
    palette="deep",
    s=100,
    alpha=1.0,
    ax=ax
)
for i, row in credit_risk_data.iterrows():
    ax.text(row[f"{prefix}Tỷ lệ nợ xấu"] + 0.1, row[f"{prefix}Tỷ lệ sử dụng vốn"], row["Quỹ"], fontsize=9)
ax.set_title("Phân loại rủi ro tín dụng")
ax.set_xlabel("Tỷ lệ nợ xấu (%)")
ax.set_ylabel("Tỷ lệ sử dụng vốn (%)")
ax.legend(title="Mức rủi ro")
st.pyplot(fig)

# Phần nhập liệu để dự đoán
st.subheader("Dự đoán rủi ro tín dụng cho Quỹ mới")
st.write("Nhập thông tin Quỹ để dự đoán mức độ rủi ro tín dụng:")

with st.form(key=f"{prefix}prediction_form"):
    total_loan = st.number_input("Tổng dư nợ (tỷ VND)", min_value=0.0, value=100.0)
    bad_debt = st.number_input("Nợ xấu (tỷ VND)", min_value=0.0, value=0.0)
    total_deposit = st.number_input("Tổng tiền gửi (tỷ VND)", min_value=0.0, value=100.0)
    submit_button = st.form_submit_button(label="Dự đoán")

if submit_button:
    # Tính toán đặc trưng từ dữ liệu nhập
    input_data = pd.DataFrame({
        "Tổng dư nợ": [total_loan],
        "Nợ xấu": [bad_debt],
        "Tổng tiền gửi": [total_deposit]
    })
    input_data[f"{prefix}Tỷ lệ nợ xấu"] = input_data["Nợ xấu"] / input_data["Tổng dư nợ"] * 100
    input_data[f"{prefix}Tỷ lệ sử dụng vốn"] = input_data["Tổng dư nợ"] / input_data["Tổng tiền gửi"] * 100

    # Chuẩn hóa dữ liệu nhập
    input_scaled = scaler.transform(input_data[features])

    # Dự đoán
    prediction = model.predict(input_scaled)
    risk_level = {0: "Thấp", 1: "Trung bình", 2: "Cao"}[prediction[0]]

    # Hiển thị kết quả dự đoán
    st.write("### Kết quả dự đoán")
    st.write(f"**Mức độ rủi ro tín dụng**: {risk_level}")
    st.write("**Thông tin đầu vào:**")
    st.write(input_data[features])

# Tải xuống kết quả
csv = credit_risk_data.to_csv(index=False)
st.download_button(
    label="Tải xuống kết quả",
    data=csv,
    file_name=f"{prefix}result.csv",
    mime="text/csv",
    key=f"{prefix}download"
)