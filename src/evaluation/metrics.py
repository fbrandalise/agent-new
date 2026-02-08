"""DeepEval metrics for product attribute enrichment evaluation."""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def get_evaluation_metrics(model: str = "gpt-4o-mini") -> list:
    """Returns DeepEval metrics tailored for product attribute enrichment.

    Metrics:
        - Completude: how many expected attributes were captured
        - Precisão: correctness of attribute values
        - Formato: JSON quality and key naming consistency
    """

    completeness = GEval(
        name="Completude",
        criteria=(
            "Avalie a completude do enriquecimento de atributos do produto. "
            "Compare os atributos gerados (actual output) com os atributos "
            "esperados (expected output). Considere: "
            "1) Quantos atributos esperados foram incluídos? "
            "2) Os atributos adicionais são relevantes para a categoria? "
            "3) Há atributos importantes faltando?"
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
    )

    accuracy = GEval(
        name="Precisão",
        criteria=(
            "Avalie a precisão dos atributos enriquecidos do produto. "
            "Compare os valores gerados com os valores esperados. Considere: "
            "1) Os valores dos atributos estão corretos? "
            "2) As unidades de medida estão corretas? "
            "3) Os valores são realistas para o produto? "
            "4) Há informações inventadas ou alucinadas?"
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
    )

    format_quality = GEval(
        name="Formato",
        criteria=(
            "Avalie a qualidade do formato dos atributos enriquecidos. "
            "Considere: "
            "1) A saída é um JSON válido e bem estruturado? "
            "2) As chaves seguem um padrão consistente (snake_case)? "
            "3) Os valores estão formatados de maneira padronizada? "
            "4) O formato facilita o uso em um sistema de e-commerce?"
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=model,
    )

    return [completeness, accuracy, format_quality]
