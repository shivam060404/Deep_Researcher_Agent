from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
## ToolNode import removed; use llm.bind_tools(tools) directly
from pydantic import BaseModel
from typing import List
from typing_extensions import TypedDict
from .tools import tools
from langchain.schema import OutputParserException

llm = ChatMistralAI()
## tool_node removed; use llm.bind_tools(tools) directly in agent functions

class ResearchPlan(BaseModel):
    steps: List[str]
    queries: List[str]

# --- New agent: UserClarification ---
class UserClarification:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a research scoping agent. Given a user request, determine if clarification is needed. If so, output a 'clarification_needed' boolean and a 'clarification_question' string. If not, output 'clarification_needed': false and 'clarified_request': the improved request.
            """
        )
        self.chain = llm | self.prompt

    def __call__(self, state):
        user_request = state["user_request"]
        result = self.chain.invoke({"input": user_request})
        # Expect result to be a dict with clarification_needed, clarification_question, clarified_request
        return result

# --- New agent: BriefGeneration ---
class BriefGeneration:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a research brief generator. Given a clarified user request, synthesize a detailed research brief describing the scope, objectives, and constraints. Output a 'brief' string.
            """
        )
        self.chain = llm | self.prompt

    def __call__(self, state):
        clarified_request = state.get("clarified_request", state["user_request"])
        result = self.chain.invoke({"input": clarified_request})
        return {"brief": result.get("brief", str(result))}

class Supervisor:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a research supervisor. Given a user request, create a step-by-step research plan and a list of search queries. Output a valid ResearchPlan object with 'steps' and 'queries'.
            """
        )
        self.chain = llm | self.prompt

    def __call__(self, state):
        # Use the brief if available, else fallback to clarified_request or user_request
        research_input = state.get("brief") or state.get("clarified_request") or state["user_request"]
        plan = self.chain.invoke({"input": research_input})
        return {"plan": plan.steps, "queries": plan.queries, "current_task_index": 0}

def research_agent(state):
    query = state["queries"][state["current_task_index"]]
    llm_with_tools = llm.bind_tools(tools)
    result = llm_with_tools.invoke({"input": query})
    return {"results": str(result)}

def writer_agent(state):
    user_request = state["user_request"]
    all_results = state["all_results"]
    prompt = ChatPromptTemplate.from_template(
        """
        You are a research writer. Write a comprehensive report for the following request: {user_request}\nUse the following findings: {all_results}\nOutput a detailed, well-structured report.
        """
    )
    chain = llm | prompt
    report = chain.invoke({"user_request": user_request, "all_results": all_results})
    return {"report": str(report)}
