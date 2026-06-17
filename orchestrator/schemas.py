from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class TaskParameters(BaseModel):
    intent: Optional[str] = Field(default=None, description="Task intent")
    planned_courses: List[str] = Field(default_factory=list, description="Course codes to check")
    course_code: Optional[str] = Field(default=None, description="Single course code")
    course_name: Optional[str] = Field(default=None, description="Course name mentioned by user")
    group_name: Optional[str] = Field(default=None, description="Course group name")
    student_id: Optional[str] = Field(default=None, description="Student id if explicitly mentioned")
    query: Optional[str] = Field(default=None, description="Worker query")
    original_question: Optional[str] = Field(default=None, description="Original user question")
    passed_courses: List[str] = Field(default_factory=list, description="Courses the student has passed")
    cumulative_gpa: Optional[Union[float, str]] = Field(default=None, description="Student cumulative GPA")


class Task(BaseModel):
    task_id: str = Field(description="Unique task id, for example TASK_01")
    task_type: Literal["RAG", "GRAG"] = Field(description="Worker type")
    query_intent: str = Field(
        description="Standalone rewritten sub-question for retrieval and synthesis"
    )
    parameters: TaskParameters = Field(
        default_factory=TaskParameters,
        description="Extracted entities and intent for the selected worker",
    )


class TaskPlan(BaseModel):
    session_id: str
    tasks: List[Task]


class CriticIssue(BaseModel):
    type: Literal[
        "missing_sub_answer",
        "unsupported_claim",
        "contradiction",
        "missing_required_context",
        "low_clarity",
        "unsafe_overconfidence",
        "other",
    ] = Field(description="Quality issue category")
    severity: Literal["low", "medium", "high"] = Field(description="Issue severity")
    target: str = Field(description="Affected task_id or final_answer")
    message: str = Field(description="Human-readable issue explanation")
    revision_instruction: str = Field(description="Concrete instruction for revising the answer")


class CriticReport(BaseModel):
    passed: bool = Field(description="Whether the answer satisfies the rubric")
    score: float = Field(description="Quality score from 0.0 to 1.0")
    issues: List[CriticIssue] = Field(default_factory=list)
    revised_answer: Optional[str] = Field(
        default=None,
        description="Corrected answer when passed is false; null or empty when no revision is needed",
    )
    error: Optional[str] = Field(default=None, description="Internal critic fallback marker")
