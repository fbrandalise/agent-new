"""LangGraph orchestration: wires the three agents into a cyclic graph."""

from langgraph.graph import END, START, StateGraph

from .agents.evaluator import evaluator_node
from .agents.runner import runner_node
from .agents.suggester import suggester_node
from .state import OrchestratorState


def _should_continue(state: OrchestratorState) -> str:
    """Return 'continue' while below max_iterations, else 'end'."""
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    return "continue"


def build_graph():
    """Build and compile the 3-agent orchestration graph.

    Flow:
        START -> evaluator -> suggester -> runner -+-> evaluator  (loop)
                                                   +-> END
    """

    builder = StateGraph(OrchestratorState)

    # Nodes
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("suggester", suggester_node)
    builder.add_node("runner", runner_node)

    # Edges
    builder.add_edge(START, "evaluator")
    builder.add_edge("evaluator", "suggester")
    builder.add_edge("suggester", "runner")

    # Conditional loop
    builder.add_conditional_edges(
        "runner",
        _should_continue,
        {"continue": "evaluator", "end": END},
    )

    return builder.compile()
