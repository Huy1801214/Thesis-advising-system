from langchain_core.prompts import ChatPromptTemplate

class Answer:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "Bạn là chuyên gia tư vấn quy chế học vụ tại Đại học Nông Lâm. "
                "Sử dụng thông tin trích dẫn dưới đây để trả lời câu hỏi của sinh viên.\n\n"
                "TRÍCH DẪN QUY CHẾ:\n{context}"
            )),
            ("human", "{input}"),
        ])

    def generate(self, query: str, context: str):
        # Tạo chuỗi tin nhắn từ template
        messages = self.prompt_template.format_messages(
            context=context, 
            input=query
        )
        # Gọi Groq LLM
        response = self.llm.invoke(messages)
        return response.content
