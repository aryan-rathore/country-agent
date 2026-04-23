from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import identify_intent, invoke_tool, synthesize_answer


def build_graph() -> StateGraph:
    """
    Build and compile the country agent graph.

    Flow:
      identify_intent → invoke_tool → synthesize_answer → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("identify_intent", identify_intent)
    graph.add_node("invoke_tool", invoke_tool)
    graph.add_node("synthesize_answer", synthesize_answer)

    # Define edges (linear pipeline)
    graph.set_entry_point("identify_intent")
    graph.add_edge("identify_intent", "invoke_tool")
    graph.add_edge("invoke_tool", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    return graph.compile()


# Single compiled instance (reused across requests — production best practice)
country_agent = build_graph()