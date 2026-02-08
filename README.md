# Orquestracao de Agentes — Avaliador de LLM

Prova de conceito de orquestracao de agentes usando **LangGraph** + **DeepEval** para avaliar e otimizar automaticamente prompts de enriquecimento de fichas tecnicas de produtos.

## Arquitetura

```
START
  |
  v
┌──────────────────────────────┐
│  Agente 1 — Avaliador        │  Enriquece atributos do produto com o LLM
│  (DeepEval: GEval metrics)   │  e avalia qualidade (Completude, Precisao, Formato)
└──────────────┬───────────────┘
               v
┌──────────────────────────────┐
│  Agente 2 — Sugestor         │  Analisa os scores e propoe novas
│  (Prompt Engineering)        │  variacoes de prompt com racional
└──────────────┬───────────────┘
               v
┌──────────────────────────────┐
│  Agente 3 — Executor         │  Prepara as sugestoes como input
│  (Runner)                    │  para o proximo ciclo
└──────────────┬───────────────┘
               |
          ┌────┴────┐
          │ loop?   │──> sim ──> Agente 1
          └────┬────┘
               │ nao
               v
              END
```

## Stack

| Componente       | Tecnologia       |
|-----------------|------------------|
| Orquestracao     | LangGraph        |
| LLM             | OpenAI (via LangChain) |
| Avaliacao        | DeepEval (GEval) |
| Interface        | Streamlit + Plotly |

## Setup

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API key (ou inserir na interface)
cp .env.example .env
# editar .env com sua OPENAI_API_KEY

# 3. Rodar a interface
streamlit run app.py
```

## Estrutura do Projeto

```
├── app.py                    # Interface Streamlit
├── requirements.txt
├── .env.example
└── src/
    ├── state.py              # Definicao do estado LangGraph
    ├── graph.py              # Montagem do grafo de orquestracao
    ├── agents/
    │   ├── evaluator.py      # Agente 1: enriquece + avalia
    │   ├── suggester.py      # Agente 2: sugere variacoes de prompt
    │   └── runner.py         # Agente 3: prepara proxima iteracao
    ├── evaluation/
    │   └── metrics.py        # Metricas DeepEval (GEval)
    └── data/
        └── products.py       # Produtos de exemplo + prompts iniciais
```

## Como funciona

1. **Agente 1 (Avaliador)**: Para cada combinacao prompt x produto, chama o LLM para enriquecer os atributos e avalia o resultado com 3 metricas DeepEval:
   - **Completude**: quantos atributos esperados foram capturados
   - **Precisao**: acuracia dos valores dos atributos
   - **Formato**: qualidade do JSON e padronizacao

2. **Agente 2 (Sugestor)**: Recebe os scores do Agente 1, identifica pontos fracos e gera novas variacoes de prompt com racional detalhado explicando as melhorias.

3. **Agente 3 (Executor)**: Salva o historico da iteracao e prepara os novos prompts como input para o Agente 1, fechando o ciclo.

O ciclo repete N vezes (configuravel), buscando melhoria continua dos prompts.
