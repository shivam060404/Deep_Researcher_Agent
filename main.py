import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 1. Set up Environment ---
# Load API keys for Mistral AI and Tavily AI from a .env file.
load_dotenv()

# --- 2. Define Tools ---
# We'll use Tavily for our web search tool. It's great for AI agents.
tavily_tool = TavilySearch(max_results=4)
tools = [tavily_tool]
tool_executor = ToolNode(tools)

# --- 3. Define the LLM ---
# We'll use Mistral's model for its powerful reasoning capabilities.
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

# --- 4. Define Agent State ---
# This is the shared memory for our agents. It will hold the research topic,
# findings, analysis, and the final report.
class AgentState(TypedDict):
    research_topic: str
    research_data: List[dict]
    analysis: str
    report: str

# --- 5. Define Agent Nodes ---
# Each agent is a function (a node in our graph) that performs a specific task.

def researcher_agent(state: AgentState):
    """
    Researches the given topic using the Tavily web search tool.
    """
    print("--- RESEARCHER AGENT ---")
    topic = state["research_topic"]
    prompt = f"""
    You are a world-class research assistant. Your task is to conduct a comprehensive, global search for the most authoritative, diverse, and up-to-date sources on the following research topic:
    '{topic}'

    Please:
    - Identify and search for information from a wide range of reputable sources, including academic journals, government and NGO reports, major news outlets, industry whitepapers, and leading expert blogs.
    - Prioritize sources that are globally recognized, high-impact, and relevant to the topic.
    - Ensure coverage of different regions, perspectives, and recent developments.
    - Find the most relevant facts, statistics, case studies, and expert opinions.
    - Maximize the breadth and quality of sources, not just quantity.
    - Avoid low-quality, non-authoritative, or duplicate sources.
    - Return a diverse set of URLs and summaries for each source found.
    """
    messages = [HumanMessage(content=prompt)]

    # Use the LLM to decide which tool to use (in this case, Tavily search)
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    # Execute the tool call and get the results
    from langchain_core.messages import AIMessage
    tool_results = tool_executor.invoke({
        "messages": messages + [AIMessage(content=response.content, tool_calls=response.tool_calls)]
    })

    # Extract and parse Tavily results from ToolMessage content
    import json
    tool_message = tool_results["messages"][0]
    content_dict = json.loads(tool_message.content)
    results = content_dict.get("results", [])
    print("DEBUG tavily results:", results)
    research_results = "\n\n".join([f"URL: {res.get('url', '')}\nContent: {res.get('content', '')}" for res in results])
    print("--- RESEARCH COMPLETE ---")
    return {"research_data": results, "research_topic": topic}

def analyst_agent(state: AgentState):
    """
    Analyzes the research data to find key insights, trends, and patterns.
    """
    print("--- ANALYST AGENT ---")
    research_data = state["research_data"]
    topic = state["research_topic"]
    
    formatted_data = "\n\n".join([f"URL: {res['url']}\nContent: {res['content']}" for res in research_data])

    prompt = f"""
    You are an expert research analyst. Your task is to perform an in-depth, comprehensive analysis of the following research data on the topic: '{topic}'.

    Research Data:
    {formatted_data}

    Please:
    - Write a detailed, multi-section analysis (at least 800-1200 words if possible).
    - Identify and explain all key trends, patterns, and significant insights, with supporting evidence from the data.
    - Compare and contrast different viewpoints or findings if present.
    - Discuss implications, challenges, and future directions related to the topic.
    - Highlight any gaps, controversies, or open questions in the research.
    - Present the analysis in a clear, logical structure with section headings and bullet points where appropriate.
    - Ensure the output is thorough, deep, and suitable for an expert audience.
    - Do not summarize too briefly; expand on each point with examples and reasoning.
    """

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    print("--- ANALYSIS COMPLETE ---")
    return {"analysis": response.content}

def writer_agent(state: AgentState):
    """
    Writes a comprehensive report based on the research and analysis.
    """
    print("--- WRITER AGENT ---")
    analysis = state["analysis"]
    topic = state["research_topic"]
    research_data = state["research_data"]

    # Extract source URLs for citation
    sources = "\n".join(sorted(list(set(d["url"] for d in research_data))))

    prompt = f"""
    You are a professional report writer. Your task is to create a comprehensive, well-structured report
    on the topic: '{topic}'.
    
    Use the following analysis and research data to draft the report.
    
    Analysis:
    {analysis}
    
    The report should have the following sections:
    1.  **Introduction**: Briefly introduce the topic.
    2.  **Key Findings**: Present the main points and data from the research.
    3.  **In-depth Analysis**: Elaborate on the insights and trends identified in the analysis.
    4.  **Conclusion**: Summarize the report and provide a concluding thought.
    5.  **Sources**: List all the source URLs.

    Sources to cite:
    {sources}
    
    Ensure the report is clear, concise, and professionally formatted.
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    print("--- REPORT GENERATED ---")
    return {"report": response.content}

# --- 6. Build the Graph ---
# Now we wire the agents together into a workflow.

# Define the graph
workflow = StateGraph(AgentState)

# Add the nodes (our agents)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)

# Set the entry point
workflow.set_entry_point("researcher")

# Add edges to define the flow
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END) # The final node

# Compile the graph into a runnable app
app = workflow.compile()

# --- 7. Run the Pipeline ---
if __name__ == "__main__":
    topic_to_research = input("Enter the topic you want to research: ")
    print(f"🚀 Starting research pipeline for topic: '{topic_to_research}'")

    # The input to the graph is a dictionary with the initial state
    inputs = {"research_topic": topic_to_research}

    # Invoke the graph and stream the results
    for output in app.stream(inputs):
        # The stream returns the output of each node as it executes
        for key, value in output.items():
            print(f"Output from node '{key}':")
            # print(value) # Uncomment to see the full state at each step
            print("-" * 30)

    # The final state contains the complete report
    final_state = app.invoke(inputs)

    print("\n\n✅ --- PIPELINE COMPLETE --- ✅\n\n")
    print("="*50)
    print("          FINAL REPORT          ")
    print("="*50)
    print(final_state['report'])

    # Save the report to a file with UTF-8 encoding
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(final_state['report'])
    print("\n\nReport saved to research_report.md")
