import streamlit as st
import requests
import time

BACKEND_URL = "http://backend:8000"
st.set_page_config(page_title= "Thesis Advising System", page_icon="🎓", layout="wide")
st.title("🎓 Hệ thống Tư vấn Khóa luận")

if "token" not in st.session_state:
    st.session_state.token = None

if not st.session_state.token:
    with st.form("login_form"):
        st.subheader("🔑 Đăng nhập hệ thống")
        username = st.text_input("MSSV")
        password = st.text_input("Mật khẩu", type="password")
        submit = st.form_submit_button("Đăng nhập")
        
        if submit:
            res = requests.post(
                f"{BACKEND_URL}/auth/login",
                data={"username": username, "password": password}
            )
            if res.status_code == 200:
                st.session_state.token = res.json()["access_token"]
                st.success("Đăng nhập thành công!")
                st.rerun()
            else:
                st.error("Sai tài khoản hoặc mật khẩu!")
else:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # --- SIDEBAR: QUẢN LÝ THÔNG TIN & UPLOAD FILE ---
    with st.sidebar:
        st.success("✅ Đã đăng nhập hệ thống!")
        
        st.markdown("---")
        st.subheader("📄 Cập nhật hồ sơ học tập")
        st.caption("Tải lên bảng điểm để AI tư vấn chính xác hơn (Chấp nhận: .csv, .xlsx)")
        
        # Widget Upload file
        uploaded_file = st.file_uploader("Chọn file bảng điểm", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("📤 Xử lý bảng điểm", use_container_width=True):
                with st.spinner("Đang trích xuất dữ liệu..."):
                    # Đóng gói file để gửi qua API (multipart/form-data)
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    
                    try:
                        res_upload = requests.post(
                            f"{BACKEND_URL}/api/grag/upload-transcript",
                            headers=headers, # Chứa token để Backend biết là sinh viên nào
                            files=files
                        )
                        
                        if res_upload.status_code == 200:
                            data = res_upload.json()
                            st.success(data["message"])
                            
                            # Hiển thị thông tin tóm tắt cho sinh viên xem
                            gpa = data.get("data", {}).get("gpa", "N/A")
                            total_passed = data.get("data", {}).get("total_passed", 0)
                            st.info(f"📊 **CPA hiện tại:** {gpa}\n\n📚 **Môn đã đậu:** {total_passed} môn")
                        else:
                            st.error(f"Lỗi: {res_upload.json().get('detail', 'Không xác định')}")
                    except Exception as e:
                        st.error(f"Không thể kết nối đến Backend: {e}")

        st.markdown("---")
        if st.button("Đăng xuất", type="primary", use_container_width=True):
            st.session_state.token = None
            st.rerun()

    # --- KHUNG CHAT CHÍNH ---
    # (Khuyên dùng: Nên lưu lịch sử chat vào st.session_state.messages để không bị mất khi ấn nút)
    if prompt := st.chat_input("Hỏi tôi về khóa luận, đăng ký tín chỉ..."):
        st.chat_message("user").markdown(prompt)
        
        # Đợi hệ thống xử lý (Loading spinner)
        with st.spinner("🤖 AI đang suy nghĩ và truy xuất đồ thị kiến thức..."):
            try:
                res_chat = requests.post(
                    f"{BACKEND_URL}/chat", 
                    params={"question": prompt}, 
                    headers=headers
                )
                
                if res_chat.status_code == 200:
                    data = res_chat.json()
                    results = data.get("data", [])
                    
                    if results:
                        with st.chat_message("assistant"):
                            st.markdown("### 💡 Phản hồi từ Hệ thống:")
                            for idx, res in enumerate(results):
                                # Trình bày đẹp hơn
                                st.info(res, icon="🎓")
                else:
                    st.error("Có lỗi xảy ra khi xử lý câu hỏi từ Backend!")
            except Exception as e:
                st.error(f"Lỗi kết nối server: {e}")