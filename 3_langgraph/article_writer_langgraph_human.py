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

def human_approval(state: ArticleState) -> ArticleState:
    print("\n" + "="*50)
    print("OUTLINE FOR REVIEW:")
    print("=" * 50)
    print(state["outline"])
    print("=" * 50)

    while True:
        approval = input("\nDo you approve this outline? (yes/no): ").lower().strip()
        if approval in ['yes','y']:
            state["approved"] = True
            print("Outline approved! Proceeding to article generation...")
            break
        elif approval in ['no','n']:
            state["approved"] = False
            print("Outline rejected. Stopping the workflow.")
            break
        else:
            print("Please enter 'yes' or 'no'")

    return state

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
builder.add_node("human_approval", human_approval)

builder.add_edge(START, "research_agent")
builder.add_edge("research_agent", "outline_agent")
builder.add_edge("outline_agent", "human_approval")
builder.add_conditional_edges("human_approval", should_contine)
builder.add_edge("writer_agent", END)

# Compile graph
graph = builder.compile()

# Run
initial_state = ArticleState(topic="Impact of AI in Education")
result = graph.invoke(initial_state)

graph.get_graph().draw_mermaid_png(output_file_path="flow_human.png")
print(graph.get_graph().print_ascii())
print("OUTLINE FOR REVIEW:")
print(f"\n*****\n{graph.get_graph().draw_mermaid()}\n*****\n")

print("\nFinal Article:\n", result["article"])

