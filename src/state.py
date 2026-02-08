"""LangGraph state definition for the agent orchestration."""

import operator
from typing import Annotated, Any, Dict, List, TypedDict


class OrchestratorState(TypedDict):
    """State shared across all agents in the orchestration graph."""

    # Input data
    products: List[Dict[str, Any]]
    current_prompts: List[Dict[str, Any]]
    model_name: str
    max_iterations: int

    # Agent outputs
    evaluation_results: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]

    # Tracking
    iteration: int
    history: Annotated[List[Dict[str, Any]], operator.add]
    logs: Annotated[List[str], operator.add]
    status: str
