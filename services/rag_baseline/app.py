import streamlit as st
import requests

# 1. Cấu hình trang web
st.set_page_config(
    page_title="Tư Vấn Học Vụ NLU",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Hệ thống Tư vấn Học vụ NLU")
st.markdown("Xin chào! Mình là trợ lý AI ảo. Bạn cần hỏi gì về quy chế, sổ tay sinh viên hay học vụ?")

# 2. Khởi tạo bộ nhớ tạm để lưu lịch sử chat trên giao diện
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Hiển thị lịch sử chat mỗi khi web reload
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📚 *Nguồn trích dẫn: {', '.join(msg['sources'])}*")

# 4. Xử lý khi người dùng nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn vào đây..."):
    # Hiển thị câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Khung hiển thị câu trả lời của AI
    with st.chat_message("assistant"):
        with st.spinner("Đang lục tìm trong quy chế..."):
            try:
                # Gọi đến API FastAPI của bạn
                api_url = "http://127.0.0.1:8000/ask"
                payload = {"question": prompt, "k": 3}
                
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    
                    # In câu trả lời ra màn hình
                    st.markdown(answer)
                    if sources:
                        st.caption(f"📚 *Nguồn trích dẫn: {', '.join(sources)}*")
                    
                    # Lưu vào lịch sử chat
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                else:
                    st.error(f"Lỗi từ hệ thống Backend (Code: {response.status_code})")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Không thể kết nối tới Backend! Bạn đã chạy file `rag_api.py` chưa?")