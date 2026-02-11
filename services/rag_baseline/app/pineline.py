from services.rag_baseline.app.llm import create_answer
from services.rag_baseline.app.prompt import build_prompt
from services.rag_baseline.app.retriever import retrive


def rag_pineline(question: str) -> str:
    context_chunk = retrive(question)
    prompt = build_prompt(context_chunk, question)
    answer = create_answer(prompt)

    return answer   