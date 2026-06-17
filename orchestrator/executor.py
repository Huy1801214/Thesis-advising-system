import asyncio
from typing import Dict, Any
from workers.rag_worker import RAGEngine
from workers.grag_worker import GRAGEngine
from workers.llm_extractor import EntityExtractor

GLOBAL_DB_SEMAPHORE = asyncio.Semaphore(20)


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
                if task.task_type == "GRAG":
                    course_name = parameters.get("course_name", "")
                    extract_target = course_name if course_name else question
                    
                    planned_codes = await self.extractor.extract_course_codes(extract_target)
                    
                    if current_intent == "registration_check":
                        parameters["planned_courses"] = planned_codes
                    elif current_intent == "course_info" and planned_codes:
                        parameters["course_code"] = planned_codes[0]
                    
                    if not student_data and current_intent in ["registration_check", "graduation_check"]:
                        msg = "Để hệ thống tư vấn chính xác về điều kiện học vụ, bạn vui lòng đính kèm file Bảng điểm (Excel/CSV) vào khung chat trước nhé!"
                        result_data["raw_data"] = f"--- Dữ liệu từ GRAG ({task.query_intent}) ---\n{msg}"
                        return result_data
                    
                    res = await self.grag_agent.query_graph(parameters)
                    result_data["raw_data"] = f"--- Dữ liệu từ GRAG ({task.query_intent}) ---\n{res}"

                elif task.task_type == "RAG":
                    res = await self.rag_agent.search_and_answer(task.query_intent) 
                    result_data["raw_data"] = f"--- Dữ liệu từ Sổ tay ({task.query_intent}) ---\n{res}"
                    
            except Exception as e:
                print(f"[error] Agent {task.task_id} ({task.task_type}) gặp sự cố: {str(e)}")
                result_data["raw_data"] = f"[Lỗi Kỹ Thuật] Không thể truy xuất thông tin phần {task.task_type}."
                result_data["status"] = "FAILED"
                result_data["error_log"] = str(e)
                
            return result_data

    async def execute_plan(self, plan: Any, question: str, mssv: str, student_data: Dict) -> Dict:
        MAX_TASKS = 5
        safe_tasks = plan.tasks[:MAX_TASKS]
        if len(plan.tasks) > MAX_TASKS:
            print(f"Warning: LLM sinh {len(plan.tasks)} tasks. Đã giảm xuống còn {MAX_TASKS} tasks an toàn.")

        # 1. Scatter: Khởi tạo danh sách các coroutines
        task_coroutines = [
            self._execute_single_task(task, question, mssv, student_data) 
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
    
