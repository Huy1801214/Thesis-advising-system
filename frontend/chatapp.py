import streamlit as st
import requests
import time

BACKEND_URL = "http://backend:8000"
st.set_page_config(page_title= "Thesis Advising System", page_icon="🎓")
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
    # --- GIAO DIỆN CHAT CHÍNH ---
    st.sidebar.success(f"Đã đăng nhập!")
    if st.sidebar.button("Đăng xuất"):
        st.session_state.token = None
        st.rerun()

    # Nhập câu hỏi
    if prompt := st.chat_input("Hỏi tôi về khóa luận..."):
        st.chat_message("user").markdown(prompt)
        
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        
        # Đợi hệ thống xử lý (Loading spinner)
        with st.spinner("🤖 AI đang suy nghĩ và truy xuất dữ liệu..."):
            # Gọi API /chat thay vì /ask
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
                        st.markdown("### Kết quả tư vấn:")
                        for idx, res in enumerate(results):
                            st.info(f"Nguồn {idx+1}: {res}")
            else:
                st.error("Có lỗi xảy ra khi kết nối với hệ thống!")