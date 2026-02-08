"""Agent 4 - Feedback Simulator: simulates a user reviewing enriched attributes."""

import src.ssl_config  # noqa: F401

import json
from typing import Any, Dict

import httpx
from langchain_openai import ChatOpenAI

from ..state import OrchestratorState

FEEDBACK_SYSTEM = (
    "Voce e um analista de qualidade de catalogo de e-commerce. "
    "Seu papel e simular o feedback de um usuario humano que revisa "
    "fichas tecnicas de produtos enriquecidas por IA. "
    "Seja critico e realista."
)

FEEDBACK_TEMPLATE = """Revise os atributos enriquecidos abaixo para o produto indicado.
Compare com os atributos esperados (ground truth) e com os atributos originais.

## Produto
- Nome: {product_name}
- Categoria: {category}
- Descricao: {description}

## Atributos originais (entrada)
{original_attributes}

## Atributos esperados (ground truth)
{expected_attributes}

## Atributos gerados pelo LLM
{enriched_output}

## Prompt utilizado
{prompt_name}

## Sua Tarefa
Para CADA atributo gerado, de um feedback:
- "positivo" se o atributo esta correto, relevante e bem formatado
- "negativo" se o atributo esta incorreto, inventado, irrelevante ou mal formatado

Responda APENAS com um JSON valido no formato:
{{
    "total_atributos": <int>,
    "positivos": <int>,
    "negativos": <int>,
    "feedbacks": [
        {{
            "atributo": "<nome do atributo>",
            "valor_gerado": "<valor que o LLM gerou>",
            "veredicto": "positivo" | "negativo",
            "motivo": "<explicacao curta>"
        }}
    ],
    "comentario_geral": "<avaliacao geral em 1-2 frases>"
}}"""


def feedback_node(state: OrchestratorState) -> Dict[str, Any]:
    """Simulates a user reviewing each enriched output and giving feedback.

    Returns feedback_results and accumulated feedback_history.
    """

    model_name = state.get("model_name", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.3,
        http_client=httpx.Client(verify=False),
    )

    evaluation_results = state["evaluation_results"]
    products = state["products"]
    iteration = state.get("iteration", 0)

    new_logs: list[str] = []
    new_logs.append("")
    new_logs.append("=" * 60)
    new_logs.append(
        f"AGENTE 4 - FEEDBACK SIMULADO  |  Iteracao {iteration + 1}"
    )
    new_logs.append("=" * 60)

    # Build product lookup
    product_map = {p["name"]: p for p in products}

    feedback_results: list[Dict[str, Any]] = []

    for result in evaluation_results:
        product = product_map.get(result["product_name"], {})

        prompt_text = FEEDBACK_TEMPLATE.replace(
            "{product_name}", product.get("name", "")
        ).replace(
            "{category}", product.get("category", "")
        ).replace(
            "{description}", product.get("description", "")
        ).replace(
            "{original_attributes}",
            json.dumps(
                product.get("attributes", {}),
                ensure_ascii=False,
                indent=2,
            ),
        ).replace(
            "{expected_attributes}",
            json.dumps(
                product.get("expected_attributes", {}),
                ensure_ascii=False,
                indent=2,
            ),
        ).replace(
            "{enriched_output}", result.get("enriched_output", "{}")
        ).replace(
            "{prompt_name}", result.get("prompt_name", "")
        )

        new_logs.append(
            f">> Revisando: {result['prompt_name']}  x  {result['product_name']}"
        )

        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": FEEDBACK_SYSTEM},
                    {"role": "user", "content": prompt_text},
                ]
            )
            response_text = response.content

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                feedback = json.loads(response_text[json_start:json_end])
            else:
                feedback = json.loads(response_text)

            positivos = feedback.get("positivos", 0)
            negativos = feedback.get("negativos", 0)
            total = feedback.get("total_atributos", positivos + negativos)

            new_logs.append(
                f"   Positivos: {positivos}  |  Negativos: {negativos}  "
                f"|  Total: {total}"
            )
            new_logs.append(
                f"   Comentario: {feedback.get('comentario_geral', '')[:150]}"
            )

        except Exception as e:
            new_logs.append(f"   ERRO no feedback: {e}")
            feedback = {
                "total_atributos": 0,
                "positivos": 0,
                "negativos": 0,
                "feedbacks": [],
                "comentario_geral": f"Erro: {e}",
            }
            positivos = 0
            negativos = 0

        feedback_results.append(
            {
                "prompt_id": result["prompt_id"],
                "prompt_name": result["prompt_name"],
                "product_name": result["product_name"],
                "positivos": positivos,
                "negativos": negativos,
                "total_atributos": feedback.get("total_atributos", 0),
                "feedbacks": feedback.get("feedbacks", []),
                "comentario_geral": feedback.get("comentario_geral", ""),
            }
        )

    # Summary per prompt
    prompt_ids_seen: list[str] = []
    for fb in feedback_results:
        if fb["prompt_id"] not in prompt_ids_seen:
            prompt_ids_seen.append(fb["prompt_id"])
    for pid in prompt_ids_seen:
        group = [fb for fb in feedback_results if fb["prompt_id"] == pid]
        total_pos = sum(fb["positivos"] for fb in group)
        total_neg = sum(fb["negativos"] for fb in group)
        pname = group[0]["prompt_name"] if group else pid
        new_logs.append("")
        new_logs.append(
            f"Resumo feedback {pname}: "
            f"+{total_pos} positivos / -{total_neg} negativos"
        )

    return {
        "feedback_results": feedback_results,
        "feedback_history": [
            {
                "iteration": iteration + 1,
                "feedbacks": feedback_results,
            }
        ],
        "logs": new_logs,
        "status": "feedback_complete",
    }
