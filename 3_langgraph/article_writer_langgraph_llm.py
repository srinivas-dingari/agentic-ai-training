from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define State schema
class ArticleState(TypedDict, total=False):
    topic: str
    research: str
    outline: str
    article: str
    approved: bool

# Define Agents
def research_agent(state: ArticleState) -> ArticleState:
    query = state["topic"]
    print("Researching on topic: ", query)
    research_output = llm.invoke(f"Research about: {query}").content
    state["research"] = research_output
    return state

def outline_agent(state: ArticleState) -> ArticleState:
    research = state["research"]
    print("Generating Outline on topic: ", state["topic"])
    outline = llm.invoke(f"Create a detailed outline based on this research: {research}").content
    state["outline"] = outline
    return state

def writer_agent(state: ArticleState) -> ArticleState:
    outline = state["outline"]
    print("Generating Article on topic: ", state["topic"])
    article = llm.invoke(f"Write a detailed article from this outline: {outline}").content
    state["article"] = article
    return state

def llm_approval(state: ArticleState) -> ArticleState:
    """LLM evaluates the outline quality and decides whether to proceed"""
    outline = state["outline"]
    topic = state["topic"]

    evaluation_prompt = f"""
    Evaluate the following outline for an artile on the topic '{topic}'
    Consider factors like:
    - Logical structure and flow
    - Completeness of coverage
    - Clarity and organization
    - Relevance to the topic
    
    Outline to evaluate:
    {outline}
    
    Respond with only 'APPROVED' if the outline id good quality and ready for article writing,
    or 'REJECTED' if it needs improvement. No other text.
    """

    decision = llm.invoke(evaluation_prompt).content.strip().upper()

    if "APPROVED" in decision:
        state["approved"] = True
        print("LLM approved the outline! Proceeding to article generation...")
    else:
        state["approved"] = False
        print("LLM rejected the outline! Stopping the workflow.")

def should_contine(state: ArticleState) -> str:
    """Conditional edge function to determine next step based on approval """
    if state.get("approved", True):
        return "writer_agent"
    else:
        return "END"

# Build LangGraph
builder = StateGraph(ArticleState)
builder.add_node("research_agent", research_agent)
builder.add_node("outline_agent", outline_agent)
builder.add_node("writer_agent", writer_agent)
builder.add_node("llm_approval", llm_approval)

builder.add_edge(START, "research_agent")
builder.add_edge("research_agent", "outline_agent")
builder.add_edge("outline_agent", "llm_approval")
builder.add_conditional_edges("llm_approval", should_contine)
builder.add_edge("writer_agent", END)

# Compile graph
graph = builder.compile()

# Run
initial_state = ArticleState(topic="Impact of AI in Education")
result = graph.invoke(initial_state)

graph.get_graph().draw_mermaid_png(output_file_path="flow_llm.png")
print(graph.get_graph().print_ascii())
print("OUTLINE FOR REVIEW:")
print(f"\n*****\n{graph.get_graph().draw_mermaid()}\n*****\n")

print("\nFinal Article:\n", result["article"])

