from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from orchestrator.critic import CriticAgent, fallback_critic_report
from orchestrator.planner import OrchestratorPlanner
from orchestrator.synthesizer import FinalSynthesizer
import uuid
from core.security import verify_token, oauth2_scheme
from api import auth
from workers.rag_worker import RAGEngine
from workers.grag_worker import GRAGEngine
from api.upload_infor import router as grag_router, STUDENT_TRANSCRIPT_STORE
from orchestrator.llm_extractor import EntityExtractor


app = FastAPI(title="Thesis Advising System API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(grag_router, prefix="/api/grag", tags=["GRAG"])

planner = OrchestratorPlanner()
synthesizer = FinalSynthesizer()
critic = CriticAgent()
extractor = EntityExtractor()
rag_agent = None
grag_agent = None


def get_rag_agent():
    global rag_agent
    if rag_agent is None:
        rag_agent = RAGEngine()
    return rag_agent


def get_grag_agent():
    global grag_agent
    if grag_agent is None:
        grag_agent = GRAGEngine()
    return grag_agent


def get_current_mssv(token: str = Depends(oauth2_scheme)):
    user_info = verify_token(token)
    return user_info.get("sub")


@app.post("/chat")
async def handle_chat(question: str, mssv: str = Depends(get_current_mssv)):
    session_id = f"SES_{mssv}_{uuid.uuid4().hex[:8]}"
    print(f"[FastAPI] Start handling question from {mssv}: {question}")

    plan = planner.create_plan(question, session_id)
    trace = []
    student_data = STUDENT_TRANSCRIPT_STORE.get(mssv, {})
    tasks = plan.tasks or []

    if not tasks:
        tasks = [
            {
                "task_id": "TASK_01",
                "task_type": "RAG",
                "query_intent": question,
                "parameters": {"intent": "policy_lookup"},
            }
        ]

    for task in tasks:
        task_id = task.task_id if hasattr(task, "task_id") else task["task_id"]
        task_type = task.task_type if hasattr(task, "task_type") else task["task_type"]
        worker_query = (
            (task.query_intent if hasattr(task, "query_intent") else task.get("query_intent"))
            or question
        ).strip()
        parameters = dict(task.parameters if hasattr(task, "parameters") else task.get("parameters", {}))

        parameters["original_question"] = question
        parameters["student_id"] = mssv
        parameters["query"] = worker_query
        parameters["passed_courses"] = student_data.get("passed_courses", [])
        parameters["cumulative_gpa"] = student_data.get("cumulative_gpa")

        print(
            mssv,
            worker_query,
            student_data.get("passed_courses", []),
            student_data.get("cumulative_gpa"),
        )
        current_intent = parameters.get("intent", "")
        trace_item = {
            "task_id": task_id,
            "worker": task_type,
            "sub_question": worker_query,
            "answer": "",
            "status": "SUCCESS",
        }

        try:
            if task_type == "GRAG" and current_intent in ["registration_check", "course_info"]:
                planned_codes = parameters.get("planned_courses") or []
                if not planned_codes:
                    planned_codes = extractor.extract_course_codes(worker_query)
                    print(f"[LLM Extractor] Extracted course codes: {planned_codes}")

                if current_intent == "registration_check":
                    parameters["planned_courses"] = planned_codes
                elif current_intent == "course_info" and planned_codes:
                    parameters["course_code"] = parameters.get("course_code") or planned_codes[0]

            if task_type == "RAG":
                print("[Agent] Querying Qdrant and Gemini (RAG)...")
                trace_item["answer"] = get_rag_agent().search_and_answer(worker_query)

            elif task_type == "GRAG":
                print("[Agent] Querying Neo4j (GRAG)...")

                if not student_data and current_intent in ["registration_check", "graduation_check"]:
                    trace_item["answer"] = (
                        "De tu van chinh xac ve dieu kien hoc vu, vui long dinh kem file "
                        "bang diem (Excel/CSV) vao khung chat truoc."
                    )
                else:
                    trace_item["answer"] = get_grag_agent().query_graph(parameters)

            else:
                trace_item["status"] = "ERROR"
                trace_item["answer"] = f"Task type khong duoc ho tro: {task_type}"

        except Exception as exc:
            print(f"[Agent] Task {task_id} failed: {exc}")
            trace_item["status"] = "ERROR"
            trace_item["answer"] = f"Khong the xu ly tac vu {task_id}: {exc}"

        trace.append(trace_item)

    if not trace:
        final_answer = "He thong chua tao duoc tac vu phu hop cho cau hoi nay."
    elif len(trace) == 1:
        final_answer = trace[0]["answer"]
    else:
        try:
            final_answer = synthesizer.synthesize(question, trace)
        except Exception as exc:
            print(f"[Synthesizer] Failed: {exc}")
            final_answer = "\n\n".join(item["answer"] for item in trace if item.get("answer"))

    try:
        critic_report = critic.review(question, final_answer, trace)
        revised_answer = (critic_report.revised_answer or "").strip()
        if not critic_report.passed and revised_answer:
            final_answer = revised_answer
    except Exception as exc:
        print(f"[Critic] Failed: {exc}")
        critic_report = fallback_critic_report()

    print("[FastAPI] Returning answer to client.")

    return {
        "session_id": session_id,
        "status": "SUCCESS",
        "answer": final_answer,
        "data": [final_answer],
        "trace": trace,
        "critic": critic_report.model_dump(),
    }
