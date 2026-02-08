"""Agent 2 - Suggester: analyses evaluation results and proposes new prompts."""

import src.ssl_config  # noqa: F401  — ensure SSL patch is active

import json
from typing import Any, Dict

import httpx
from langchain_openai import ChatOpenAI

from ..state import OrchestratorState

SUGGESTER_SYSTEM = (
    "Voce e um especialista em engenharia de prompts para LLMs, "
    "focado em otimizar prompts para enriquecimento de fichas tecnicas "
    "de produtos de e-commerce."
)

SUGGESTER_TEMPLATE = """Analise os resultados de avaliacao abaixo de prompts usados para enriquecer fichas tecnicas de produtos.

## Resultados da Avaliacao

{evaluation_summary}

## Prompts Atuais

{current_prompts}

## Sua Tarefa

Com base nos resultados, sugira {num_suggestions} novas variacoes de prompt que melhorem os scores.

Para cada sugestao:
1. Identifique os pontos fracos dos prompts atuais
2. Explique o racional da melhoria proposta (seja detalhado)
3. Forneca o template completo do novo prompt

IMPORTANTE: Os templates DEVEM usar exatamente estas variaveis de formatacao Python:
{{product_name}}, {{category}}, {{description}}, {{brand}}, {{attributes}}

Responda APENAS com um JSON valido (array) no formato:
[
    {{
        "id": "prompt_v{{N}}",
        "name": "Nome descritivo da variacao",
        "template": "O template completo do prompt usando {{product_name}}, {{category}}, {{description}}, {{brand}}, {{attributes}}",
        "rationale": "Explicacao detalhada do racional e das melhorias propostas"
    }}
]"""


def suggester_node(state: OrchestratorState) -> Dict[str, Any]:
    """Analyses evaluation scores and proposes improved prompt variations.

    Returns new suggestions and log entries.
    """

    model_name = state.get("model_name", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        http_client=httpx.Client(verify=False),
    )

    evaluation_results = state["evaluation_results"]
    current_prompts = state["current_prompts"]
    iteration = state.get("iteration", 0)

    new_logs: list[str] = []
    new_logs.append("")
    new_logs.append("=" * 60)
    new_logs.append(
        f"AGENTE 2 - SUGESTOR DE PROMPTS  |  Iteracao {iteration + 1}"
    )
    new_logs.append("=" * 60)

    # Build evaluation summary
    eval_parts: list[str] = []
    for result in evaluation_results:
        metrics_str = ", ".join(
            f"{name}: {data['score']:.2f}"
            for name, data in result["metrics"].items()
        )
        eval_parts.append(
            f"- {result['prompt_name']} x {result['product_name']}: "
            f"Score medio={result['avg_score']:.2f} | {metrics_str}"
        )
    evaluation_summary = "\n".join(eval_parts)

    # Build prompts summary
    prompts_parts: list[str] = []
    for p in current_prompts:
        prompts_parts.append(
            f"### {p['name']} (ID: {p['id']})\n"
            f"Racional: {p.get('rationale', 'N/A')}\n"
            f"Template:\n```\n{p['template']}\n```"
        )
    prompts_summary = "\n\n".join(prompts_parts)

    user_prompt = SUGGESTER_TEMPLATE.format(
        evaluation_summary=evaluation_summary,
        current_prompts=prompts_summary,
        num_suggestions=2,
    )

    new_logs.append(">> Analisando resultados e gerando sugestoes...")

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": SUGGESTER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )
        response_text = response.content

        # Extract JSON array from response
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            suggestions = json.loads(json_str)
        else:
            suggestions = json.loads(response_text)

        # Ensure unique IDs
        next_id = iteration * 2 + 3
        for i, suggestion in enumerate(suggestions):
            suggestion["id"] = f"prompt_v{next_id + i}"
            new_logs.append("")
            new_logs.append(f"Sugestao: {suggestion['name']}")
            rationale = suggestion.get("rationale", "")
            new_logs.append(f"   Racional: {rationale[:300]}")

    except Exception as e:
        new_logs.append(f"ERRO ao gerar sugestoes: {e}")
        # Fallback: keep current prompts
        suggestions = current_prompts

    return {
        "suggestions": suggestions,
        "logs": new_logs,
        "status": "suggestions_ready",
    }
