from typing_extensions import TypedDict
from typing import List, Annotated
from langgraph.graph import StateGraph
import operator
from .agents import Supervisor, research_agent, writer_agent, UserClarification, BriefGeneration

class AgentState(TypedDict):
    user_request: str
    clarification_needed: bool
    clarification_question: str
    clarified_request: str
    brief: str
    plan: List[str]
    queries: List[str]
    results: str
    all_results: Annotated[List[str], operator.add]
    report: str
    current_task_index: int

def should_continue(state: AgentState):
    if state["current_task_index"] < len(state["plan"]):
        return "continue_research"
    return "write_report"

def execute_task(state: AgentState):
    return {"all_results": [state["results"]]}

def increment_task_index(state: AgentState):
    return {"current_task_index": state["current_task_index"] + 1}

user_clarification = UserClarification()
brief_generation = BriefGeneration()
supervisor = Supervisor()
graph = StateGraph(AgentState)
graph.add_node("user_clarification", user_clarification)
graph.add_node("brief_generation", brief_generation)
graph.add_node("supervisor", supervisor)
graph.add_node("research", research_agent)
graph.add_node("execute_task", execute_task)
graph.add_node("increment_task_index", increment_task_index)
graph.add_node("writer", writer_agent)
graph.set_entry_point("user_clarification")
graph.add_edge("user_clarification", "brief_generation")
graph.add_edge("brief_generation", "supervisor")
graph.add_edge("supervisor", "research")
graph.add_edge("research", "execute_task")
graph.add_edge("execute_task", "increment_task_index")
graph.add_conditional_edges(
    "increment_task_index",
    should_continue,
    {
        "continue_research": "research",
        "write_report": "writer"
    }
)
graph.add_edge("writer", "END")
app = graph.compile()
