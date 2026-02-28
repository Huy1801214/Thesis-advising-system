import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1, 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return llm
