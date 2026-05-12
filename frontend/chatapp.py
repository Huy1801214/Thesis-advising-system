import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"
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
        
        # 1. Gọi API /ask
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        res_ask = requests.post(f"{BACKEND_URL}/ask", params={"question": prompt}, headers=headers)
        
        if res_ask.status_code == 200:
            data = res_ask.json()
            session_id = data["session_id"]
            task_ids = data["task_ids"]
            
            # 2. Cơ chế Polling (Đợi kết quả)
            with st.status("🤖 AI đang suy nghĩ và truy xuất dữ liệu...", expanded=True) as status:
                results = None
                while True:
                    # Gọi API /sync
                    res_sync = requests.get(
                        f"{BACKEND_URL}/sync/{session_id}",
                        params={"task_ids": task_ids},
                        headers=headers
                    )
                    
                    if res_sync.status_code == 200:
                        sync_data = res_sync.json()
                        if sync_data["status"] == "SUCCESS":
                            results = sync_data["data"]
                            status.update(label="✅ Đã có câu trả lời!", state="complete", expanded=False)
                            break
                        else:
                            # Nếu đang WAITING thì đợi 2 giây rồi hỏi lại
                            time.sleep(2)
                    else:
                        st.error("Lỗi khi đồng bộ dữ liệu!")
                        break
            
            # 3. Hiển thị kết quả
            if results:
                with st.chat_message("assistant"):
                    st.markdown("### Kết quả tư vấn:")
                    for idx, res in enumerate(results):
                        st.info(f"Nguồn {idx+1}: {res}")
        else:
            st.error("Có lỗi xảy ra khi gửi câu hỏi!")