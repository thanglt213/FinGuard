import streamlit as st

st.set_page_config(page_title="Giám sát Quỹ Tín dụng Nhân dân", layout="wide")

st.title("Ứng dụng Giám sát Quỹ Tín dụng Nhân dân")
st.write("""
 
Ứng dụng này hỗ trợ 5 bài toán chính:
1. **Phát hiện biến động bất thường**: Dùng Isolation Forest để tìm các tháng có biến động tài chính bất thường.
2. **Phát hiện mất khả năng thanh toán**: Dùng Isolation Forest (unsupervised) để xác định Quỹ có nguy cơ mất thanh khoản.
3. **Đánh giá mức độ rủi ro tín dụng**: Dùng Logistic Regression (đa lớp - supervised) để đánh giá mức độ rủi ro của Quỹ.
4. **Phát hiện thất thoát tài sản**: Dùng Linear Regression để dự đoán và phát hiện thất thoát tài sản.
5. **Kiểm tra tuân thủ an toàn vốn và tỷ lệ nợ xấu**

Vui lòng chọn bài toán từ menu bên trái để bắt đầu!
""")