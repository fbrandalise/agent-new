"""Agent 3 - Runner: takes suggestions and prepares the next evaluation cycle."""

from typing import Any, Dict

from ..state import OrchestratorState


def runner_node(state: OrchestratorState) -> Dict[str, Any]:
    """Saves the current iteration to history and sets up the next one.

    Returns updated current_prompts, iteration counter, and history entry.
    """

    suggestions = state["suggestions"]
    evaluation_results = state["evaluation_results"]
    current_prompts = state["current_prompts"]
    iteration = state.get("iteration", 0)

    new_logs: list[str] = []
    new_logs.append("")
    new_logs.append("=" * 60)
    new_logs.append(
        f"AGENTE 3 - EXECUTOR  |  Iteracao {iteration + 1}"
    )
    new_logs.append("=" * 60)

    feedback_results = state.get("feedback_results", [])

    # Build a history snapshot for this iteration (returned as single-item
    # list so the Annotated[..., operator.add] reducer appends it).
    history_entry = {
        "iteration": iteration + 1,
        "prompts_used": current_prompts,
        "evaluations": evaluation_results,
        "feedback": feedback_results,
        "suggestions": suggestions,
    }

    new_logs.append(f"Iteracao {iteration + 1} salva no historico")
    new_logs.append(
        f">> Preparando {len(suggestions)} prompts para proxima iteracao:"
    )
    for s in suggestions:
        new_logs.append(f"   - {s.get('name', s.get('id', '?'))}")

    next_iteration = iteration + 1

    if next_iteration >= state["max_iterations"]:
        new_logs.append("")
        new_logs.append("Limite de iteracoes atingido. Finalizando.")

    return {
        "current_prompts": suggestions,
        "evaluation_results": [],  # reset for next cycle
        "iteration": next_iteration,
        "history": [history_entry],  # single-item list for reducer
        "logs": new_logs,
        "status": "ready_for_next_iteration",
    }
