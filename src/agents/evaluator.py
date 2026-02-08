"""Agent 1 - Evaluator: enriches product attributes and evaluates with DeepEval."""

import src.ssl_config  # noqa: F401  — ensure SSL patch is active

import json
from typing import Any, Dict

import httpx
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI

from ..evaluation.metrics import get_evaluation_metrics
from ..state import OrchestratorState


def evaluator_node(state: OrchestratorState) -> Dict[str, Any]:
    """Runs each prompt x product combination, then scores with DeepEval.

    Returns new evaluation_results and log entries.
    """

    model_name = state.get("model_name", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        http_client=httpx.Client(verify=False),
    )

    products = state["products"]
    prompts = state["current_prompts"]
    iteration = state.get("iteration", 0)

    new_logs: list[str] = []
    new_logs.append("")
    new_logs.append("=" * 60)
    new_logs.append(
        f"AGENTE 1 - AVALIADOR  |  Iteracao {iteration + 1}"
    )
    new_logs.append("=" * 60)

    evaluation_results: list[Dict[str, Any]] = []
    metrics = get_evaluation_metrics(model=model_name)

    for prompt in prompts:
        for product in products:
            # -- 1. Build enrichment prompt from template --
            enrichment_prompt = prompt["template"].format(
                product_name=product["name"],
                category=product["category"],
                description=product["description"],
                brand=product["brand"],
                attributes=json.dumps(
                    product["attributes"], ensure_ascii=False, indent=2
                ),
            )

            new_logs.append("")
            new_logs.append(
                f">> {prompt['name']}  x  {product['name']}"
            )

            # -- 2. Call LLM for enrichment --
            try:
                response = llm.invoke(enrichment_prompt)
                enriched_output = response.content
            except Exception as e:
                new_logs.append(f"   ERRO na geracao: {e}")
                enriched_output = "{}"

            # -- 3. Evaluate with DeepEval metrics --
            expected = json.dumps(
                product.get("expected_attributes", {}),
                ensure_ascii=False,
                indent=2,
            )

            test_case = LLMTestCase(
                input=enrichment_prompt,
                actual_output=enriched_output,
                expected_output=expected,
                context=[product["description"]],
            )

            metric_results: Dict[str, Any] = {}
            for metric in metrics:
                try:
                    metric.measure(test_case)
                    metric_results[metric.name] = {
                        "score": metric.score,
                        "reason": metric.reason,
                    }
                    new_logs.append(
                        f"   {metric.name}: {metric.score:.2f}"
                    )
                except Exception as e:
                    metric_results[metric.name] = {
                        "score": 0.0,
                        "reason": f"Erro: {e}",
                    }
                    new_logs.append(
                        f"   {metric.name}: erro - {e}"
                    )

            scores = [
                m["score"]
                for m in metric_results.values()
                if isinstance(m["score"], (int, float))
            ]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            evaluation_results.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt_name": prompt["name"],
                    "product_name": product["name"],
                    "enriched_output": enriched_output,
                    "metrics": metric_results,
                    "avg_score": avg_score,
                }
            )

            new_logs.append(f"   Score medio: {avg_score:.2f}")

    # Per-prompt summary
    for prompt in prompts:
        prompt_results = [
            r for r in evaluation_results if r["prompt_id"] == prompt["id"]
        ]
        if prompt_results:
            avg = sum(r["avg_score"] for r in prompt_results) / len(
                prompt_results
            )
            new_logs.append("")
            new_logs.append(
                f"Resumo {prompt['name']}: Score medio = {avg:.2f}"
            )

    return {
        "evaluation_results": evaluation_results,
        "logs": new_logs,
        "status": "evaluation_complete",
    }
