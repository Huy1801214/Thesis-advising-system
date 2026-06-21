import asyncio
from typing import Dict, Any
from workers.rag_worker import RAGEngine
from workers.grag_worker import GRAGEngine
from workers.llm_extractor import EntityExtractor

GLOBAL_DB_SEMAPHORE = asyncio.Semaphore(20)
TASK_TIMEOUT = 15.0

def task_parameters_to_dict(parameters: Any) -> Dict[str, Any]:
    if parameters is None:
        return {}
    if hasattr(parameters, "model_dump"):
        return parameters.model_dump(exclude_none=True)
    return dict(parameters)

class TaskExecutor:
    def __init__(self):
        self.rag_agent = RAGEngine()
        self.grag_agent = GRAGEngine()
        self.extractor = EntityExtractor()

    async def _execute_with_retry(self, func, *args, max_retries=2, **kwargs):
        last_exception = None
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except asyncio.TimeoutError as e:
                last_exception = e
                print(f"[Warning] Mất kết nối DB lần {attempt + 1}. Đang thử lại...")
            except Exception as e:
                last_exception = e
                print(f"[Warning] Lỗi truy xuất lần {attempt + 1}: {str(e)}")
                await asyncio.sleep(1) 
        raise last_exception
    
    async def _execute_single_task(self, task: Any, question: str, mssv: str, student_data: Dict) -> Dict:        
        async with GLOBAL_DB_SEMAPHORE:
            parameters = task_parameters_to_dict(getattr(task, "parameters", None))
                
            parameters.update({
                "student_id": mssv,
                "query": task.query_intent,
                "passed_courses": student_data.get("passed_courses", []) if student_data else [],
                "cumulative_gpa": student_data.get("cumulative_gpa") if student_data else None
            })
            
            current_intent = parameters.get("intent", "")
            result_data = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "query_intent": task.query_intent,
                "raw_data": "", 
                "status": "SUCCESS"
            }
            
            try:
                if task.task_type == "CLARIFY":
                    clarify_msg = parameters.get("clarification_message", "Hệ thống cần thêm thông tin từ bạn.")
                    result_data["raw_data"] = f"--- [Yêu cầu làm rõ từ hệ thống] ---\n{clarify_msg}"
                    result_data["status"] = "CLARIFYING"
                    return result_data
                
                elif task.task_type == "GRAG":
                    course_name = parameters.get("course_name", "")
                    extract_target = course_name if course_name else question
                    
                    planned_codes = await self.extractor.extract_course_codes(extract_target)
                    
                    if current_intent == "registration_check":
                        parameters["planned_courses"] = planned_codes
                    elif current_intent == "course_info" and planned_codes:
                        parameters["course_code"] = planned_codes[0]
                    
                    if not student_data and current_intent in ["registration_check", "graduation_check"]:
                        msg = "Để hệ thống tư vấn chính xác về điều kiện học vụ, bạn vui lòng đính kèm file Bảng điểm (Excel/CSV) vào khung chat trước nhé!"
                        result_data["raw_data"] = f"--- Dữ liệu từ GRAG ({task.query_intent}) ---\n[THIẾU BẢNG ĐIỂM] {msg}"
                        result_data["status"] = "MISSING_DATA"
                        return result_data
                    
                    res = await self._execute_with_retry(self.grag_agent.query_graph, parameters)
                    result_data["raw_data"] = f"--- Dữ liệu từ GRAG ({task.query_intent}) ---\n{res}"

                elif task.task_type == "RAG":
                    res = await self._execute_with_retry(self.rag_agent.search_and_answer, task.query_intent) 
                    result_data["raw_data"] = f"--- Dữ liệu từ Sổ tay ({task.query_intent}) ---\n{res}"
                    
            except asyncio.TimeoutError:
                result_data["raw_data"] = f"[Error] Agent {task.task_type} quá thời gian xử lý."
                result_data["status"] = "TIMEOUT"
            except Exception as e:
                result_data["raw_data"] = f"[Error] Agent {task.task_type} gặp sự cố nội bộ: {str(e)}"
                result_data["status"] = "FAILED"
                result_data["error_log"] = str(e)
                
            return result_data

    async def execute_plan(self, plan: Any, question: str, mssv: str, student_data: Dict) -> Dict:
        if getattr(plan, "needs_clarification", False) or any(t.task_type == "CLARIFY" for t in plan.tasks):
            clarify_task = next((t for t in plan.tasks if t.task_type == "CLARIFY"), None)
            msg = clarify_task.parameters.clarification_message if clarify_task else "Xin vui lòng cung cấp thêm chi tiết."
            print("[Executor] Dừng truy vấn DB. Yêu cầu sinh viên làm rõ câu hỏi.")
            return {
                "raw_context": f"--- [THÔNG ĐIỆP TỪ HỆ THỐNG] ---\n{msg}",
                "debug_data": [],
                "tasks_execution_info": [{"task_id": "SYS_CLARIFY", "status": "CLARIFYING", "message": msg}]
            }
        
        MAX_TASKS = 5
        safe_tasks = plan.tasks[:MAX_TASKS]
        if len(plan.tasks) > MAX_TASKS:
            print(f"Warning: LLM sinh {len(plan.tasks)} tasks. Đã giảm xuống còn {MAX_TASKS} tasks an toàn.")

        # 1. Scatter: Khởi tạo danh sách các coroutines
        task_coroutines = [
            asyncio.wait_for(
                self._execute_single_task(task, question, mssv, student_data),
                timeout=TASK_TIMEOUT
            )
            for task in safe_tasks
        ]
        
        # 2. Gather: Thực thi đồng thời trên Event Loop
        executed_tasks = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        raw_results = []
        debug_tasks_info = []
        
        for result in executed_tasks:
            if isinstance(result, Exception):
                print(f"[Fatal Error] Lỗi nghiêm trọng ngoài tầm kiểm soát: {str(result)}")
                continue
            raw_results.append(result["raw_data"])
            debug_tasks_info.append(result)

        # 3. Đóng gói Raw Context 
        print("[TaskExecutor] Đã gom đủ dữ liệu thô. Chuyển giao cho Generator...")
        
        if raw_results:
            combined_context = "\n\n".join(raw_results)
        else:
            combined_context = "Hệ thống không tìm thấy thông tin cơ sở dữ liệu phù hợp cho truy vấn này."
        
        print("\n" + "🔥"*25)
        print("🎯 [DEBUG] RAW CONTEXT TỪ CÁC AGENT TRẢ VỀ:")
        print("-" * 50)
        print(combined_context)
        print("🔥"*25 + "\n")

        return {
            "raw_context": combined_context,
            "debug_data": raw_results,
            "tasks_execution_info": debug_tasks_info
        }
    
