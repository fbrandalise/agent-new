# Orquestracao de Agentes para Avaliacao de LLMs
## Documento Tecnico para Lideranca Executiva

**Data:** Fevereiro 2026
**Status:** Prova de Conceito (PoC)

---

## 1. Resumo Executivo

Esta prova de conceito demonstra um sistema de **otimizacao automatica de prompts** para enriquecimento de fichas tecnicas de produtos usando inteligencia artificial.

O problema que resolve: quando usamos um LLM (Large Language Model) para enriquecer dados de catalogo de produtos, a qualidade do resultado depende diretamente da qualidade do prompt — a instrucao que damos ao modelo. Hoje, otimizar esses prompts e um processo manual, lento e dependente de tentativa e erro.

**O que esta PoC faz:** coloca quatro agentes de IA trabalhando em ciclo para avaliar, coletar feedback simulado de usuario, sugerir melhorias e re-testar prompts de forma autonoma, buscando melhoria continua com reforco humano simulado.

**Resultado esperado:** reducao significativa do tempo de otimizacao de prompts e aumento mensuravel na qualidade do enriquecimento de dados de produto, guiado por feedback atributo a atributo.

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
- **Sem feedback estruturado** — nao ha mecanismo para capturar o que o usuario aprovaria ou rejeitaria
- **Nao escalavel** — cada categoria de produto pode precisar de prompts diferentes

### O que esta PoC propoe
Automatizar o ciclo inteiro de avaliacao e otimizacao usando agentes autonomos que:
1. Medem qualidade com metricas objetivas
2. Simulam feedback de um usuario revisando cada atributo gerado
3. Sugerem melhorias baseadas nos resultados E no feedback
4. Testam as melhorias automaticamente
5. Acumulam reforcos positivos e negativos para guiar novas rodadas

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
│              AGENTE 4 — FEEDBACK SIMULADO                    │
│                                                              │
│  Simula um usuario humano revisando cada atributo gerado:    │
│  1. Compara cada atributo com o ground truth                 │
│  2. Da veredicto "positivo" ou "negativo" por atributo       │
│  3. Justifica cada decisao (ex: "valor inventado",           │
│     "unidade incorreta", "atributo irrelevante")             │
│  4. Acumula contagem de reforcos (+/-) no historico          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│              AGENTE 2 — SUGESTOR                             │
│                                                              │
│  Recebe os scores do Agente 1 E o feedback do Agente 4:     │
│  1. Identifica pontos fracos de cada prompt                  │
│  2. Presta atencao especial nos atributos NEGATIVOS          │
│  3. Propoe 2 novas variacoes com racional detalhado          │
│  4. Explica POR QUE cada mudanca deve corrigir os problemas  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│              AGENTE 3 — EXECUTOR                             │
│                                                              │
│  1. Salva resultados, feedback e sugestoes no historico       │
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
  (novo ciclo)                 comparativos,
                               feedback acumulado
                               e vencedor
```

### Como os agentes se comunicam

Os quatro agentes compartilham um **estado unico** (state) gerenciado pelo LangGraph. Cada agente le o que precisa do estado, executa sua logica e devolve suas atualizacoes. Nao ha chamadas diretas entre agentes — toda comunicacao passa pelo estado centralizado.

```
┌──────────────────────────────────────────┐
│          ESTADO COMPARTILHADO             │
│                                           │
│  • produtos (entrada fixa)                │
│  • prompts atuais (evolui a cada          │
│    iteracao)                              │
│  • resultados da avaliacao                │
│  • feedback por atributo (+/-)            │
│  • historico de feedback acumulado        │
│  • sugestoes do sugestor                  │
│  • historico completo                     │
│  • logs de execucao                       │
│  • iteracao atual / maximo                │
└──────────────────────────────────────────┘
       ▲         ▲         ▲         ▲
       │         │         │         │
   Agente 1  Agente 4  Agente 2  Agente 3
  (avaliador)(feedback)(sugestor)(executor)
```

### O papel do feedback na melhoria continua

O Agente 4 (Feedback Simulado) introduz um mecanismo de **reforco** no ciclo. Para cada atributo gerado, ele decide se o atributo e aceitavel (+) ou problematico (-), com justificativa. Esses reforcos:

1. **Informam o Agente 2**: os atributos negativos viram exemplos concretos de problemas a corrigir nos novos prompts
2. **Se acumulam ao longo das iteracoes**: o historico de feedback cresce, permitindo medir se a taxa de aprovacao evolui
3. **Fornecem metricas complementares**: enquanto DeepEval mede qualidade de forma automatica, o feedback simula a perspectiva do usuario final

---

## 4. Stack Tecnologica

### Tabela de tecnologias

| Camada             | Tecnologia         | O que faz                                                     | Por que foi escolhida                                                |
|--------------------|--------------------|---------------------------------------------------------------|----------------------------------------------------------------------|
| **Orquestracao**   | LangGraph          | Define o grafo de agentes, controla fluxo e loop              | Framework da LangChain para agentes com estado; padrao de mercado    |
| **LLM**           | OpenAI (GPT-4o)    | Executa o enriquecimento, feedback e sugestoes                | API mais madura, suporte a JSON mode, custo/beneficio                |
| **Avaliacao**      | DeepEval           | Mede qualidade do output com metricas customizaveis           | Framework open-source, gratuito, nativo para avaliacao de LLMs       |
| **Interface**      | Streamlit          | Dashboard interativo com execucao em tempo real               | Prototipagem rapida em Python, sem necessidade de frontend separado  |
| **Visualizacao**   | Plotly             | Graficos comparativos (barras, radar, linha)                  | Graficos interativos, integrado com Streamlit                        |
| **Integracao LLM** | LangChain          | Wrapper padronizado para chamadas ao LLM                      | Abstrai provider (OpenAI, Azure, Anthropic), facilita troca futura   |

### Detalhamento das tecnologias-chave

#### LangGraph — Orquestracao de Agentes
LangGraph e um framework da LangChain especificamente projetado para construir fluxos de agentes com **estado persistente** e **ciclos** (loops). Diferentemente de pipelines lineares, ele permite que agentes revisitem etapas anteriores — exatamente o que precisamos para o ciclo de melhoria continua.

Conceitos principais:
- **StateGraph**: grafo onde cada no e um agente e as arestas definem o fluxo
- **State**: dicionario tipado compartilhado entre todos os agentes
- **Conditional Edges**: decisoes de roteamento (continuar o loop ou finalizar)
- **Streaming**: permite acompanhar a execucao em tempo real na interface

#### DeepEval — Avaliacao de Qualidade
DeepEval e um framework open-source para avaliacao de LLMs. Usamos o modulo **GEval** que permite definir criterios de avaliacao em linguagem natural e usa um LLM para pontuar o resultado de 0 a 1.

Metricas configuradas nesta PoC:

| Metrica        | O que mede                                          | Exemplo de score baixo                           |
|----------------|-----------------------------------------------------|--------------------------------------------------|
| **Completude** | % de atributos esperados que foram preenchidos       | Retornou 3 de 15 atributos esperados             |
| **Precisao**   | Correcao factual dos valores preenchidos             | Disse que Galaxy S24 tem "Android 12" (incorreto) |
| **Formato**    | Qualidade estrutural do JSON de saida                | Chaves inconsistentes, JSON mal formado          |

#### Feedback Simulado — Reforco por Atributo
O Agente 4 complementa as metricas automaticas com uma **perspectiva de usuario**. Para cada atributo gerado:

| Veredicto   | Criterio                                                              | Exemplo                                        |
|-------------|-----------------------------------------------------------------------|-------------------------------------------------|
| **Positivo** | Atributo correto, relevante e bem formatado                          | "memoria_ram": "8GB" — correto                  |
| **Negativo** | Atributo incorreto, inventado, irrelevante ou com formato inadequado | "processador": "Exynos 2400" — valor incorreto  |

Os reforcos negativos sao repassados como exemplos concretos ao Agente 2, que os usa para gerar prompts que corrijam esses problemas especificos.

#### Streamlit — Interface Visual
Streamlit permite construir dashboards interativos escrevendo apenas Python. A interface desta PoC possui:

- **Tab Produtos**: visualizacao dos produtos de teste e seus atributos esperados (ground truth)
- **Tab Prompts**: visualizacao das variacoes de prompt iniciais
- **Tab Execucao**: acompanhamento em tempo real com cards por agente, incluindo feedback, e painel de log colorido
- **Tab Resultados**: graficos comparativos, feedback acumulado, timeline de prompts, avaliacao de vencedor e exportacao JSON

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
    ├── ssl_config.py             Configuracao de SSL (ambiente corporativo)
    ├── state.py                  Definicao do estado compartilhado
    ├── graph.py                  Montagem do grafo LangGraph (4 agentes)
    │
    ├── agents/
    │   ├── evaluator.py          Agente 1: enriquecimento + avaliacao
    │   ├── feedback.py           Agente 4: feedback simulado por atributo
    │   ├── suggester.py          Agente 2: sugestao de novos prompts
    │   └── runner.py             Agente 3: controle de ciclo
    │
    ├── evaluation/
    │   └── metrics.py            Configuracao das metricas DeepEval
    │
    └── data/
        └── products.py           Produtos de exemplo e prompts iniciais
```

**Total: ~1.700 linhas de codigo Python** distribuidas em 11 arquivos com responsabilidades claras e separadas.

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

### Iteracao 1 — Agente 4 da feedback
O agente revisa cada atributo gerado simulando um usuario humano:

| Prompt               | Positivos | Negativos | Taxa de Aprovacao |
|----------------------|-----------|-----------|-------------------|
| Prompt Simples (v1)  | +8        | -4        | 67%               |
| Prompt Estruturado (v2) | +11    | -2        | 85%               |

Exemplos de feedback negativo:
- `processador: "Exynos 2400"` — NEGATIVO: valor incorreto, Galaxy S24 usa Snapdragon 8 Gen 3
- `peso: "N/A"` — NEGATIVO: valor generico, deveria ser "167g"

### Iteracao 1 — Agente 2 sugere
Com base nos scores E nos feedbacks negativos, identifica que:
- v1 tem formato ruim (0.55) e 4 atributos negativos — sugere adicionar instrucao de JSON e evitar valores genericos
- v2 pode melhorar precisao — sugere incluir instrucao para nao inventar valores desconhecidos
- Gera dois novos prompts (v3, v4) com racional detalhado

### Iteracao 1 — Agente 3 executa
Salva tudo (avaliacao + feedback + sugestoes) no historico e prepara v3 e v4 para a proxima iteracao.

### Iteracao 2 — Ciclo repete
Agente 1 avalia v3 e v4 → Agente 4 revisa → Agente 2 sugere v5 e v6 → Agente 3 finaliza.

### Resultado final
Dashboard com:
- Graficos comparando scores de todas as variacoes
- Evolucao da taxa de aprovacao do feedback (ex: 67% → 85% → 92%)
- Ranking completo com vencedor e melhoria percentual vs prompt inicial
- Historico atributo a atributo de cada iteracao

---

## 7. Custos Estimados de Execucao

### Por execucao da PoC (2 iteracoes, 3 produtos, 2 prompts)

| Componente                | Chamadas ao LLM | Tokens estimados | Custo estimado (GPT-4o-mini) |
|---------------------------|-----------------|------------------|------------------------------|
| Enriquecimento (Agente 1) | 12 chamadas     | ~24.000 tokens   | ~US$ 0.004                   |
| Avaliacao DeepEval         | 36 chamadas     | ~72.000 tokens   | ~US$ 0.011                   |
| Feedback Simulado (Agente 4) | 12 chamadas  | ~36.000 tokens   | ~US$ 0.005                   |
| Sugestao (Agente 2)       | 2 chamadas      | ~10.000 tokens   | ~US$ 0.002                   |
| **Total**                 | **~62 chamadas** | **~142.000 tokens** | **~US$ 0.022**            |

**Nota:** Usando `gpt-4o-mini` o custo e praticamente desprezivel. Com `gpt-4o` o custo sobe ~20x mas ainda se mantem em centavos por execucao.

---

## 8. Diferenciais da Abordagem

| Aspecto              | Abordagem Manual (atual)            | Esta PoC (agentes autonomos)                 |
|----------------------|-------------------------------------|----------------------------------------------|
| **Tempo por ciclo**  | Horas a dias                        | Minutos                                      |
| **Objetividade**     | Avaliacao subjetiva                 | Scores numericos reprodutiveis (0-1)         |
| **Feedback**         | Revisao informal, sem registro      | Feedback atributo a atributo com justificativa, acumulado no historico |
| **Rastreabilidade**  | Sem historico padronizado           | Historico completo com racional de cada mudanca |
| **Escalabilidade**   | 1 engenheiro = 1 teste por vez      | Multiplos produtos e prompts em paralelo     |
| **Transparencia**    | "Esse prompt parece melhor"         | "Score subiu de 0.63 para 0.83, aprovacao de 67% para 92%, porque..." |
| **Aprendizado**      | Nao se acumula entre ciclos         | Reforcos positivos/negativos guiam cada nova rodada |

---

## 9. Mecanismo de Feedback e Reforco

### Como funciona o ciclo de reforco

```
Iteracao 1:  +8 positivos / -4 negativos  → Taxa: 67%
                                              │
                    Agente 2 recebe os 4      │
                    negativos como exemplos    │
                    concretos a corrigir       │
                                              v
Iteracao 2:  +11 positivos / -2 negativos → Taxa: 85%
                                              │
                    Agente 2 recebe os 2      │
                    negativos restantes        │
                                              v
Iteracao 3:  +12 positivos / -1 negativo  → Taxa: 92%
```

### O que e registrado para cada atributo

| Campo           | Descricao                              | Exemplo                           |
|-----------------|----------------------------------------|-----------------------------------|
| **atributo**    | Nome do atributo avaliado              | processador                       |
| **valor_gerado** | O que o LLM gerou                    | Exynos 2400                       |
| **veredicto**   | positivo ou negativo                   | negativo                          |
| **motivo**      | Justificativa do veredicto             | Valor incorreto para Galaxy S24   |

### Metricas de feedback na interface

A aba de Resultados mostra:
- **KPIs**: total de avaliacoes, reforcos positivos, negativos e taxa de aprovacao
- **Grafico de barras**: positivos vs negativos por iteracao
- **Grafico de linha**: evolucao da taxa de aprovacao ao longo das iteracoes
- **Detalhamento**: feedback atributo a atributo com veredicto colorido e motivo

---

## 10. Avaliacao de Vencedor

Ao final de todas as iteracoes, o sistema automaticamente:

1. **Identifica o vencedor**: o prompt com maior score medio entre todas as iteracoes
2. **Mostra o ranking completo**: todos os prompts ordenados por score com medalhas
3. **Calcula a melhoria**: delta absoluto e percentual comparado ao melhor prompt inicial
4. **Exibe o template vencedor**: racional e template completo prontos para uso em producao

---

## 11. Limitacoes da PoC e Proximos Passos

### Limitacoes atuais
- **Dados de teste limitados:** 3 produtos pre-cadastrados com ground truth manual
- **Sem integracao com catalogo real:** os dados nao vem de um sistema de PIM/catalogo
- **Modelo unico:** usa apenas OpenAI; nao compara com outros providers (Anthropic, Google, etc.)
- **Avaliacao usa o proprio LLM:** as metricas DeepEval (GEval) usam um LLM para avaliar outro LLM, o que introduz um vies circular
- **Feedback simulado:** o Agente 4 e uma simulacao via LLM, nao um usuario humano real

### Evolucoes sugeridas

| Prioridade | Evolucao                                    | Impacto                                     |
|------------|---------------------------------------------|---------------------------------------------|
| Alta       | Integrar com catalogo real (API ou CSV)     | Validacao com dados reais de producao        |
| Alta       | Incluir metricas deterministicas            | Reduzir dependencia do LLM para avaliacao    |
| Alta       | Substituir feedback simulado por feedback humano real (RLHF) | Validacao genuina de qualidade |
| Media      | Suporte multi-provider (Anthropic, Gemini)  | Comparar custo/qualidade entre modelos       |
| Media      | Persistencia de resultados (banco de dados) | Historico entre sessoes e acumulo de feedback |
| Baixa      | Pipeline CI/CD para rodar automaticamente   | Avaliacao continua em producao               |
| Baixa      | A/B testing de prompts em producao          | Validar melhorias com usuarios reais         |

---

## 12. Como Rodar a PoC

### Pre-requisitos
- Python 3.10+
- Chave de API da OpenAI

### Comandos
```bash
pip install -r requirements.txt
streamlit run app.py
```

A chave da OpenAI pode ser configurada via arquivo `.env` ou diretamente na interface web.
O numero de iteracoes (ate 10) e configuravel na barra lateral.

---

## 13. Glossario

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
| **Feedback simulado** | Avaliacao atributo a atributo feita por um LLM simulando a perspectiva de um usuario humano |
| **Reforco (+/-)**  | Contagem acumulada de aprovacoes e rejeicoes de atributos ao longo das iteracoes               |
| **RLHF**          | Reinforcement Learning from Human Feedback — tecnica de aprendizado com feedback humano real    |
| **Taxa de aprovacao** | Percentual de atributos que receberam reforco positivo em relacao ao total avaliado          |
