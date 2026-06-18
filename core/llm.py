import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def build_chat_model(temperature: float = 0, max_retries: int = 3) -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )
