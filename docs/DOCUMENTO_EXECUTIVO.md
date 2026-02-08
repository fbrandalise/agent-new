# Orquestracao de Agentes para Avaliacao de LLMs
## Documento Tecnico para Lideranca Executiva

**Data:** Fevereiro 2026
**Status:** Prova de Conceito (PoC)

---

## 1. Resumo Executivo

Esta prova de conceito demonstra um sistema de **otimizacao automatica de prompts** para enriquecimento de fichas tecnicas de produtos usando inteligencia artificial.

O problema que resolve: quando usamos um LLM (Large Language Model) para enriquecer dados de catalogo de produtos, a qualidade do resultado depende diretamente da qualidade do prompt — a instrucao que damos ao modelo. Hoje, otimizar esses prompts e um processo manual, lento e dependente de tentativa e erro.

**O que esta PoC faz:** coloca tres agentes de IA trabalhando em ciclo para avaliar, sugerir melhorias e re-testar prompts de forma autonoma, buscando melhoria continua sem intervencao humana.

**Resultado esperado:** reducao significativa do tempo de otimizacao de prompts e aumento mensuravel na qualidade do enriquecimento de dados de produto.

---

## 2. O Problema de Negocio

### Contexto
Fichas tecnicas de produtos em e-commerce frequentemente chegam incompletas dos fornecedores. Um celular pode vir apenas com "256GB Preto" quando o consumidor precisa saber RAM, tamanho de tela, tipo de camera, bateria, etc.

### Como LLMs ajudam
Modelos de linguagem conseguem inferir e completar esses atributos a partir de descricoes basicas. Porem, a qualidade desse enriquecimento varia drasticamente conforme o **prompt** utilizado.

### O gargalo atual
Otimizar prompts e um processo:
- **Manual** — engenheiros testam variacoes uma a uma
- **Subjetivo** — sem metricas padronizadas de qualidade
- **Lento** — cada iteracao exige revisao humana
- **Nao escalavel** — cada categoria de produto pode precisar de prompts diferentes

### O que esta PoC propoe
Automatizar o ciclo inteiro de avaliacao e otimizacao usando agentes autonomos que:
1. Medem qualidade com metricas objetivas
2. Sugerem melhorias baseadas nos resultados
3. Testam as melhorias automaticamente
4. Repetem o ciclo ate atingir o limite configurado

---

## 3. Arquitetura da Solucao

### Visao Geral do Fluxo

```
┌─────────────────────────────────────────────────────────────┐
│                    INICIO DA EXECUCAO                        │
│         (produtos + 2 variacoes de prompt iniciais)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│              AGENTE 1 — AVALIADOR                            │
│                                                              │
│  Para cada combinacao prompt x produto:                      │
│  1. Chama o LLM para enriquecer os atributos                │
│  2. Compara o resultado com o esperado (ground truth)        │
│  3. Gera scores de 0 a 1 em tres dimensoes:                 │
│     • Completude — quantos atributos foram preenchidos?      │
│     • Precisao — os valores estao corretos?                  │
│     • Formato — o JSON esta bem estruturado?                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│              AGENTE 2 — SUGESTOR                             │
│                                                              │
│  Recebe os scores do Agente 1 e:                             │
│  1. Identifica pontos fracos de cada prompt                  │
│  2. Propoe 2 novas variacoes com racional detalhado          │
│  3. Explica POR QUE cada mudanca deve melhorar o score       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│              AGENTE 3 — EXECUTOR                             │
│                                                              │
│  1. Salva os resultados da iteracao no historico             │
│  2. Prepara os novos prompts como entrada                    │
│  3. Decide: rodar mais um ciclo ou finalizar?                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                ┌─────┴─────┐
                │  Mais      │
                │ iteracoes? │
                └─────┬─────┘
                 sim  │  nao
                  │   │
    ┌─────────────┘   └──────────────┐
    v                                v
  AGENTE 1                     FIM: Resultados
  (novo ciclo)                 comparativos e
                               historico completo
```

### Como os agentes se comunicam

Os tres agentes compartilham um **estado unico** (state) gerenciado pelo LangGraph. Cada agente le o que precisa do estado, executa sua logica e devolve suas atualizacoes. Nao ha chamadas diretas entre agentes — toda comunicacao passa pelo estado centralizado.

```
┌─────────────────────────────────────┐
│          ESTADO COMPARTILHADO        │
│                                      │
│  • produtos (entrada fixa)           │
│  • prompts atuais (evolui a cada     │
│    iteracao)                         │
│  • resultados da avaliacao           │
│  • sugestoes do sugestor             │
│  • historico completo                │
│  • logs de execucao                  │
│  • iteracao atual / maximo           │
└─────────────────────────────────────┘
       ▲            ▲            ▲
       │            │            │
   Agente 1     Agente 2     Agente 3
   (le e        (le e        (le e
    escreve)     escreve)     escreve)
```

---

## 4. Stack Tecnologica

### Tabela de tecnologias

| Camada             | Tecnologia         | O que faz                                                     | Por que foi escolhida                                                |
|--------------------|--------------------|---------------------------------------------------------------|----------------------------------------------------------------------|
| **Orquestracao**   | LangGraph          | Define o grafo de agentes, controla fluxo e loop              | Framework da LangChain para agentes com estado; padrao de mercado    |
| **LLM**           | OpenAI (GPT-4o)    | Executa o enriquecimento e gera sugestoes de prompt           | API mais madura, suporte a JSON mode, custo/beneficio                |
| **Avaliacao**      | DeepEval           | Mede qualidade do output com metricas customizaveis           | Framework open-source, gratuito, nativo para avaliacao de LLMs       |
| **Interface**      | Streamlit          | Dashboard interativo com execucao em tempo real               | Prototipagem rapida em Python, sem necessidade de frontend separado  |
| **Visualizacao**   | Plotly             | Graficos comparativos (barras e radar)                        | Graficos interativos, integrado com Streamlit                        |
| **Integracao LLM** | LangChain          | Wrapper padronizado para chamadas ao LLM                      | Abstrai provider (OpenAI, Azure, Anthropic), facilita troca futura   |

### Detalhamento das tecnologias-chave

#### LangGraph — Orquestracao de Agentes
LangGraph e um framework da LangChain especificamente projetado para construir fluxos de agentes com **estado persistente** e **ciclos** (loops). Diferentemente de pipelines lineares, ele permite que agentes revisitem etapas anteriores — exatamente o que precisamos para o ciclo de melhoria continua.

Conceitos principais:
- **StateGraph**: grafo onde cada no e um agente e as arestas definem o fluxo
- **State**: dicionario tipado compartilhado entre todos os agentes
- **Conditional Edges**: decisoes de roteamento (continuar o loop ou finalizar)
- **Streaming**: permite acompanhar a execucao no em tempo real na interface

#### DeepEval — Avaliacao de Qualidade
DeepEval e um framework open-source para avaliacao de LLMs. Usamos o modulo **GEval** que permite definir criterios de avaliacao em linguagem natural e usa um LLM para pontuar o resultado de 0 a 1.

Metricas configuradas nesta PoC:

| Metrica        | O que mede                                          | Exemplo de score baixo                           |
|----------------|-----------------------------------------------------|--------------------------------------------------|
| **Completude** | % de atributos esperados que foram preenchidos       | Retornou 3 de 15 atributos esperados             |
| **Precisao**   | Correcao factual dos valores preenchidos             | Disse que Galaxy S24 tem "Android 12" (incorreto) |
| **Formato**    | Qualidade estrutural do JSON de saida                | Chaves inconsistentes, JSON mal formado          |

#### Streamlit — Interface Visual
Streamlit permite construir dashboards interativos escrevendo apenas Python. A interface desta PoC possui:

- **Tab Produtos**: visualizacao dos produtos de teste e seus atributos esperados (ground truth)
- **Tab Prompts**: visualizacao das variacoes de prompt iniciais
- **Tab Execucao**: acompanhamento em tempo real com cards por agente e painel de log colorido
- **Tab Resultados**: graficos comparativos, historico detalhado e exportacao JSON

---

## 5. Estrutura do Codigo

```
projeto/
│
├── app.py                        Interface Streamlit (ponto de entrada)
│
├── requirements.txt              Dependencias Python
│
└── src/
    ├── state.py                  Definicao do estado compartilhado
    ├── graph.py                  Montagem do grafo LangGraph
    │
    ├── agents/
    │   ├── evaluator.py          Agente 1: enriquecimento + avaliacao
    │   ├── suggester.py          Agente 2: sugestao de novos prompts
    │   └── runner.py             Agente 3: controle de ciclo
    │
    ├── evaluation/
    │   └── metrics.py            Configuracao das metricas DeepEval
    │
    └── data/
        └── products.py           Produtos de exemplo e prompts iniciais
```

**Total: ~1.200 linhas de codigo Python** distribuidas em 10 arquivos com responsabilidades claras e separadas.

---

## 6. Exemplo Pratico: Fluxo de uma Execucao

### Entrada
**Produto:** Smartphone Samsung Galaxy S24 256GB Preto
**Atributos iniciais:** cor: "Preto", armazenamento: "256GB"
**Prompt v1 (simples):** "Enriqueca a ficha tecnica com todos os atributos relevantes"
**Prompt v2 (estruturado):** Prompt com instrucoes detalhadas, regras e formato de saida

### Iteracao 1 — Agente 1 avalia
O agente roda cada prompt no LLM e compara o resultado com os 15 atributos esperados:

| Prompt               | Completude | Precisao | Formato | Media |
|----------------------|-----------|----------|---------|-------|
| Prompt Simples (v1)  | 0.65      | 0.70     | 0.55    | 0.63  |
| Prompt Estruturado (v2) | 0.80   | 0.82     | 0.88    | 0.83  |

### Iteracao 1 — Agente 2 sugere
Com base nos scores, identifica que:
- v1 tem formato ruim (0.55) — sugere adicionar instrucao explicita de JSON com snake_case
- v2 pode melhorar completude (0.80) — sugere adicionar lista de categorias de atributos a serem preenchidos
- Gera dois novos prompts (v3, v4) com racional detalhado

### Iteracao 1 — Agente 3 executa
Salva tudo no historico e prepara v3 e v4 como entrada para a proxima iteracao.

### Iteracao 2 — Ciclo repete
Agente 1 avalia v3 e v4 → Agente 2 sugere v5 e v6 → Agente 3 finaliza.

### Resultado final
Dashboard com graficos comparando todas as variacoes, mostrando a evolucao dos scores e o racional de cada mudanca.

---

## 7. Custos Estimados de Execucao

### Por execucao da PoC (2 iteracoes, 3 produtos, 2 prompts)

| Componente                | Chamadas ao LLM | Tokens estimados | Custo estimado (GPT-4o-mini) |
|---------------------------|-----------------|------------------|------------------------------|
| Enriquecimento (Agente 1) | 12 chamadas     | ~24.000 tokens   | ~US$ 0.004                   |
| Avaliacao DeepEval         | 36 chamadas     | ~72.000 tokens   | ~US$ 0.011                   |
| Sugestao (Agente 2)       | 2 chamadas      | ~8.000 tokens    | ~US$ 0.001                   |
| **Total**                 | **~50 chamadas** | **~104.000 tokens** | **~US$ 0.016**            |

**Nota:** Usando `gpt-4o-mini` o custo e praticamente desprezivel. Com `gpt-4o` o custo sobe ~20x mas ainda se mantem em centavos por execucao.

---

## 8. Diferenciais da Abordagem

| Aspecto              | Abordagem Manual (atual)            | Esta PoC (agentes autonomos)                 |
|----------------------|-------------------------------------|----------------------------------------------|
| **Tempo por ciclo**  | Horas a dias                        | Minutos                                      |
| **Objetividade**     | Avaliacao subjetiva                 | Scores numericos reprodutiveis (0-1)         |
| **Rastreabilidade**  | Sem historico padronizado           | Historico completo com racional de cada mudanca |
| **Escalabilidade**   | 1 engenheiro = 1 teste por vez      | Multiplos produtos e prompts em paralelo     |
| **Transparencia**    | "Esse prompt parece melhor"         | "Score subiu de 0.63 para 0.83 porque..."    |

---

## 9. Limitacoes da PoC e Proximos Passos

### Limitacoes atuais
- **Dados de teste limitados:** 3 produtos pre-cadastrados com ground truth manual
- **Sem integracao com catalogo real:** os dados nao vem de um sistema de PIM/catalogo
- **Modelo unico:** usa apenas OpenAI; nao compara com outros providers (Anthropic, Google, etc.)
- **Avaliacao usa o proprio LLM:** as metricas DeepEval (GEval) usam um LLM para avaliar outro LLM, o que introduz um vies circular

### Evolucoes sugeridas

| Prioridade | Evolucao                                    | Impacto                                     |
|------------|---------------------------------------------|---------------------------------------------|
| Alta       | Integrar com catalogo real (API ou CSV)     | Validacao com dados reais de producao        |
| Alta       | Incluir metricas deterministicas            | Reduzir dependencia do LLM para avaliacao    |
| Media      | Suporte multi-provider (Anthropic, Gemini)  | Comparar custo/qualidade entre modelos       |
| Media      | Persistencia de resultados (banco de dados) | Historico entre sessoes                      |
| Baixa      | Pipeline CI/CD para rodar automaticamente   | Avaliacao continua em producao               |
| Baixa      | A/B testing de prompts em producao          | Validar melhorias com usuarios reais         |

---

## 10. Como Rodar a PoC

### Pre-requisitos
- Python 3.10+
- Chave de API da OpenAI

### Comandos
```bash
pip install -r requirements.txt
streamlit run app.py
```

A chave da OpenAI pode ser configurada via arquivo `.env` ou diretamente na interface web.

---

## 11. Glossario

| Termo              | Definicao                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------|
| **LLM**           | Large Language Model — modelo de linguagem de grande escala (ex: GPT-4, Claude)                 |
| **Prompt**         | Instrucao/texto enviado ao LLM para gerar uma resposta                                        |
| **Enriquecimento** | Processo de completar atributos faltantes de um produto a partir de dados minimos               |
| **Ficha tecnica**  | Conjunto de atributos estruturados de um produto (cor, peso, dimensoes, etc.)                  |
| **Ground truth**   | Resultado esperado/correto usado como referencia para avaliacao                                |
| **Agente**         | Componente autonomo de software que executa uma tarefa especifica dentro de um fluxo maior     |
| **GEval**          | Metrica do DeepEval que usa um LLM para avaliar a saida de outro LLM com base em criterios customizados |
| **LangGraph**      | Framework para construcao de grafos de agentes com estado e ciclos                             |
| **Streamlit**      | Framework Python para construcao rapida de dashboards web interativos                          |
| **DeepEval**       | Framework open-source para avaliacao automatizada de LLMs                                      |
| **Orquestracao**   | Coordenacao automatica de multiplos agentes em um fluxo de trabalho definido                   |
