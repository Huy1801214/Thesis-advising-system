import streamlit as st
import requests
import time

BACKEND_URL = "http://backend:8000"
st.set_page_config(page_title= "Thesis Advising System", page_icon="🎓", layout="wide")
st.title("🎓 Hệ thống Tư vấn Khóa luận")

if "token" not in st.session_state:
    st.session_state.token = None
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            st.session_state.messages = []
            st.rerun()

    # --- KHUNG CHAT CHÍNH ---
    # (Khuyên dùng: Nên lưu lịch sử chat vào st.session_state.messages để không bị mất khi ấn nút)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hỏi tôi về khóa luận, đăng ký tín chỉ..."):
        st.chat_message("user").markdown(prompt)
        
        # Đợi hệ thống xử lý (Loading spinner)
        with st.spinner("🤖 Orchestrator đang phân rã tác vụ và truy xuất đồ thị..."):
            try:
                res_chat = requests.post(
                    f"{BACKEND_URL}/chat", 
                    params={"question": prompt}, 
                    headers=headers
                )
                
                if res_chat.status_code == 200:
                    data = res_chat.json()
                    
                    # Lấy dữ liệu từ Backend theo đúng API Contract mới
                    final_answer = data.get("answer", "Xin lỗi, hệ thống không thể tạo câu trả lời.")
                    critic_score = data.get("critic_score")
                    trace_data = data.get("debug_trace", [])
                    is_clarifying = any(t.get("status") == "CLARIFYING" for t in trace_data)
                    # Hiển thị tin nhắn của AI
                    with st.chat_message("assistant"):
                        if is_clarifying:
                            st.warning(f"**Hệ thống cần thêm thông tin:**\n\n{final_answer}", icon="⚠️")
                        else:
                            st.markdown(final_answer)
                        
                        # --- VŨ KHÍ BẢO VỆ ĐỒ ÁN: SHOW LUỒNG SUY LUẬN ---
                        if trace_data:
                            with st.expander("🛠️ Chi tiết luồng xử lý Multi-Agent (Dành cho Giám khảo)"):
                                if critic_score is not None:
                                    st.markdown(f"**🏅 Điểm đánh giá (Critic Score):** `{critic_score}/1.0`")
                                
                                st.markdown("**Tiến trình thực thi song song:**")
                                for t in trace_data:
                                    status_icon = "✅" if t.get('status') == "SUCCESS" else "❌"
                                    st.caption(f"{status_icon} **[{t.get('task_type')}]** {t.get('query_intent')}")
                        
                    # Lưu lại vào bộ nhớ lịch sử
                    st.session_state.messages.append({"role": "assistant", "content": final_answer}, "is_clarify": is_clarifying)
                else:
                    st.error(f"Lỗi {res_chat.status_code}: {res_chat.text}")
            except Exception as e:
                st.error(f"Lỗi kết nối server: {e}. Hãy đảm bảo FastAPI đang chạy ở {BACKEND_URL}")