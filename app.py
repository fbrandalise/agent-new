"""Streamlit visual interface for the LLM Evaluator Agent Orchestration."""

import src.ssl_config  # noqa: F401  — must be first to patch SSL globally

import json
import os
import time

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data.products import INITIAL_PROMPTS, SAMPLE_PRODUCTS  # noqa: E402
from src.graph import build_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Orquestração de Agentes - Avaliador de LLM",
    page_icon="🔄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .agent-card {
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .agent-evaluator { background: linear-gradient(135deg, #667eea22, #764ba222); border-left: 4px solid #667eea; }
    .agent-suggester { background: linear-gradient(135deg, #f093fb22, #f5576c22); border-left: 4px solid #f093fb; }
    .agent-runner    { background: linear-gradient(135deg, #4facfe22, #00f2fe22); border-left: 4px solid #4facfe; }
    .metric-box {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin: 4px;
    }
    .score-high { color: #28a745; font-weight: bold; font-size: 1.4em; }
    .score-mid  { color: #ffc107; font-weight: bold; font-size: 1.4em; }
    .score-low  { color: #dc3545; font-weight: bold; font-size: 1.4em; }
    .iteration-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 16px 0 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Configuracao")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Necessaria para LLM e avaliacao DeepEval",
    )

    model_name = st.selectbox(
        "Modelo",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
        index=0,
    )

    max_iterations = st.slider(
        "Iteracoes de otimizacao",
        min_value=1,
        max_value=10,
        value=2,
        help="Quantas vezes o ciclo avaliacao->sugestao->execucao roda",
    )

    st.divider()
    st.subheader("Produtos de teste")
    selected_products = st.multiselect(
        "Selecione os produtos",
        options=[p["name"] for p in SAMPLE_PRODUCTS],
        default=[p["name"] for p in SAMPLE_PRODUCTS],
    )

    st.divider()
    st.markdown(
        "**Fluxo do grafo**\n\n"
        "```\n"
        "START\n"
        "  |-> Agente 1: Avaliador\n"
        "  |-> Agente 2: Sugestor\n"
        "  |-> Agente 3: Executor\n"
        "  |-> (loop ou END)\n"
        "```"
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Orquestracao de Agentes — Avaliador de LLM")
st.markdown(
    "Avaliacao automatica de qualidade de enriquecimento de fichas tecnicas "
    "de produtos usando **LangGraph** + **DeepEval**."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_products, tab_prompts, tab_run, tab_results = st.tabs(
    ["Produtos", "Prompts Iniciais", "Execucao", "Resultados"]
)

# ---- Tab: Produtos --------------------------------------------------------
with tab_products:
    st.subheader("Produtos de Exemplo")
    for product in SAMPLE_PRODUCTS:
        if product["name"] not in selected_products:
            continue
        with st.expander(f"{product['name']}  ({product['brand']})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dados basicos**")
                st.json(
                    {
                        "nome": product["name"],
                        "categoria": product["category"],
                        "descricao": product["description"],
                        "marca": product["brand"],
                        "atributos_iniciais": product["attributes"],
                    }
                )
            with col2:
                st.markdown("**Atributos esperados (ground truth)**")
                st.json(product["expected_attributes"])

# ---- Tab: Prompts ---------------------------------------------------------
with tab_prompts:
    st.subheader("Variacoes de Prompt Iniciais")
    for prompt in INITIAL_PROMPTS:
        with st.expander(f"{prompt['name']}"):
            st.markdown(f"**Racional:** {prompt['rationale']}")
            st.code(prompt["template"], language="text")

# ---------------------------------------------------------------------------
# Helper functions for display
# ---------------------------------------------------------------------------


def _score_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    if score >= 0.4:
        return "score-mid"
    return "score-low"


def _render_evaluation_results(results: list) -> None:
    """Render evaluation results as metric cards."""
    if not results:
        return

    for result in results:
        with st.container():
            st.markdown(
                f"**{result['prompt_name']}** × **{result['product_name']}**"
            )
            cols = st.columns(len(result["metrics"]) + 1)

            for i, (metric_name, data) in enumerate(
                result["metrics"].items()
            ):
                score = data["score"]
                css = _score_class(score)
                with cols[i]:
                    st.markdown(
                        f"<div class='metric-box'>"
                        f"<div style='font-size:0.85em'>{metric_name}</div>"
                        f"<div class='{css}'>{score:.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            with cols[-1]:
                avg = result["avg_score"]
                css = _score_class(avg)
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div style='font-size:0.85em'>Media</div>"
                    f"<div class='{css}'>{avg:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("Ver output enriquecido"):
                st.code(result["enriched_output"], language="json")

            st.divider()


def _render_suggestions(suggestions: list) -> None:
    """Render prompt suggestions with rationale."""
    for s in suggestions:
        st.markdown(f"**{s.get('name', s.get('id'))}**")
        st.info(f"Racional: {s.get('rationale', 'N/A')}")
        with st.expander("Ver template completo"):
            st.code(s.get("template", ""), language="text")


def _build_comparison_chart(history: list) -> go.Figure:
    """Build a Plotly chart comparing prompt scores across iterations."""
    fig = go.Figure()

    for entry in history:
        iteration = entry["iteration"]
        # Group by prompt
        prompt_scores: dict[str, list[float]] = {}
        for ev in entry["evaluations"]:
            pid = ev["prompt_name"]
            prompt_scores.setdefault(pid, []).append(ev["avg_score"])

        for prompt_name, scores in prompt_scores.items():
            avg = sum(scores) / len(scores) if scores else 0
            fig.add_trace(
                go.Bar(
                    name=f"Iter {iteration}: {prompt_name}",
                    x=[f"Iteracao {iteration}"],
                    y=[avg],
                    text=[f"{avg:.2f}"],
                    textposition="auto",
                )
            )

    fig.update_layout(
        title="Evolucao dos Scores por Iteracao",
        yaxis_title="Score Medio",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=400,
    )
    return fig


def _build_metrics_radar(history: list) -> go.Figure:
    """Build a radar chart for the latest iteration's metric breakdown."""
    if not history:
        return go.Figure()

    latest = history[-1]
    evaluations = latest["evaluations"]

    fig = go.Figure()

    # Group by prompt
    prompt_groups: dict[str, list] = {}
    for ev in evaluations:
        prompt_groups.setdefault(ev["prompt_name"], []).append(ev)

    for prompt_name, evals in prompt_groups.items():
        # Average each metric across products
        metric_names: list[str] = []
        metric_avgs: list[float] = []
        if evals and evals[0].get("metrics"):
            for m_name in evals[0]["metrics"]:
                metric_names.append(m_name)
                vals = [
                    e["metrics"][m_name]["score"]
                    for e in evals
                    if m_name in e.get("metrics", {})
                ]
                metric_avgs.append(
                    sum(vals) / len(vals) if vals else 0
                )

        fig.add_trace(
            go.Scatterpolar(
                r=metric_avgs + [metric_avgs[0]] if metric_avgs else [],
                theta=metric_names + [metric_names[0]]
                if metric_names
                else [],
                fill="toself",
                name=prompt_name,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar de Metricas (ultima iteracao)",
        height=400,
    )
    return fig


# ---- Tab: Execution -------------------------------------------------------
with tab_run:
    st.subheader("Executar Orquestracao")

    if not api_key:
        st.warning("Insira sua OpenAI API Key na barra lateral para iniciar.")

    run_button = st.button(
        "Iniciar Orquestracao",
        disabled=not api_key,
        type="primary",
        use_container_width=True,
    )

    if run_button and api_key:
        os.environ["OPENAI_API_KEY"] = api_key

        products = [
            p for p in SAMPLE_PRODUCTS if p["name"] in selected_products
        ]
        if not products:
            st.error("Selecione ao menos um produto na barra lateral.")
            st.stop()

        initial_state = {
            "products": products,
            "current_prompts": INITIAL_PROMPTS,
            "evaluation_results": [],
            "suggestions": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "history": [],
            "logs": [],
            "status": "starting",
            "model_name": model_name,
        }

        graph = build_graph()

        # Layout: left = agent cards, right = live log
        col_agents, col_log = st.columns([3, 2])

        with col_log:
            st.markdown("**Log de Execucao**")
            log_container = st.container(height=520)

        # Progress tracking
        with col_agents:
            progress_bar = st.progress(0)
            status_text = st.empty()
        step = 0
        total_steps = max_iterations * 3

        # Containers for each iteration
        all_logs: list[str] = []
        full_history: list = []
        last_eval_results: list = []

        def _append_logs(new_lines: list[str]) -> None:
            """Append log lines and refresh the live log panel."""
            all_logs.extend(new_lines)
            with log_container:
                for line in new_lines:
                    if line.startswith("="):
                        st.markdown(
                            f"<span style='color:#667eea;font-weight:bold'>"
                            f"{line}</span>",
                            unsafe_allow_html=True,
                        )
                    elif "ERRO" in line:
                        st.markdown(
                            f"<span style='color:#dc3545'>{line}</span>",
                            unsafe_allow_html=True,
                        )
                    elif "Score medio" in line or "Resumo" in line:
                        st.markdown(
                            f"<span style='color:#28a745'>{line}</span>",
                            unsafe_allow_html=True,
                        )
                    elif line.startswith(">>") or line.startswith("Sugestao"):
                        st.markdown(
                            f"<span style='color:#f093fb'>{line}</span>",
                            unsafe_allow_html=True,
                        )
                    elif line.strip():
                        st.text(line)

        for event in graph.stream(
            initial_state, stream_mode="updates"
        ):
            for node_name, updates in event.items():
                step += 1
                pct = min(step / total_steps, 1.0)
                progress_bar.progress(pct)

                new_logs = updates.get("logs", [])
                _append_logs(new_logs)

                # --- Evaluator ---
                if node_name == "evaluator":
                    status_text.markdown(
                        "**Agente 1 — Avaliador** executando..."
                    )
                    eval_results = updates.get("evaluation_results", [])
                    last_eval_results = eval_results

                    with col_agents:
                        st.markdown(
                            "<div class='agent-card agent-evaluator'>"
                            "<strong>Agente 1 — Avaliador</strong></div>",
                            unsafe_allow_html=True,
                        )
                        _render_evaluation_results(eval_results)

                # --- Suggester ---
                elif node_name == "suggester":
                    status_text.markdown(
                        "**Agente 2 — Sugestor** gerando variacoes..."
                    )
                    suggestions = updates.get("suggestions", [])

                    with col_agents:
                        st.markdown(
                            "<div class='agent-card agent-suggester'>"
                            "<strong>Agente 2 — Sugestor de Prompts"
                            "</strong></div>",
                            unsafe_allow_html=True,
                        )
                        _render_suggestions(suggestions)

                # --- Runner ---
                elif node_name == "runner":
                    status_text.markdown(
                        "**Agente 3 — Executor** preparando proxima iteracao..."
                    )
                    new_history = updates.get("history", [])
                    full_history.extend(new_history)

                    iteration_num = updates.get("iteration", 0)
                    with col_agents:
                        st.markdown(
                            "<div class='agent-card agent-runner'>"
                            f"<strong>Agente 3 — Executor</strong> "
                            f"| Iteracao {iteration_num} concluida</div>",
                            unsafe_allow_html=True,
                        )

                        if iteration_num < max_iterations:
                            st.markdown("Iniciando proxima iteracao...")
                        else:
                            st.markdown("Todas as iteracoes concluidas.")

                        st.divider()

        progress_bar.progress(1.0)
        status_text.markdown("**Orquestracao concluida!**")

        # Store results in session state for the Results tab
        st.session_state["history"] = full_history
        st.session_state["all_logs"] = all_logs


# ---- Tab: Results ----------------------------------------------------------
with tab_results:
    st.subheader("Resultados Comparativos")

    history = st.session_state.get("history", [])
    if not history:
        st.info("Execute a orquestracao primeiro para ver os resultados.")
    else:
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_bar = _build_comparison_chart(history)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_chart2:
            fig_radar = _build_metrics_radar(history)
            st.plotly_chart(fig_radar, use_container_width=True)

        # Detailed iteration history
        st.divider()
        st.subheader("Historico por Iteracao")
        for entry in history:
            it = entry["iteration"]
            with st.expander(f"Iteracao {it}", expanded=(it == len(history))):
                st.markdown("**Prompts utilizados:**")
                for p in entry.get("prompts_used", []):
                    st.markdown(f"- {p['name']}: {p.get('rationale', '')}")

                st.markdown("**Avaliacoes:**")
                _render_evaluation_results(entry.get("evaluations", []))

                st.markdown("**Sugestoes geradas:**")
                _render_suggestions(entry.get("suggestions", []))

        # Evolution table
        st.divider()
        st.subheader("Evolucao dos Prompts")
        for i, entry in enumerate(history):
            cols = st.columns([1, 4])
            with cols[0]:
                st.metric("Iteracao", entry["iteration"])
            with cols[1]:
                for p in entry.get("prompts_used", []):
                    st.markdown(f"**{p['name']}**")
                    st.caption(p.get("rationale", ""))

        # Raw logs
        all_logs = st.session_state.get("all_logs", [])
        if all_logs:
            with st.expander("Logs Completos"):
                st.code("\n".join(all_logs), language="text")

        # Export
        st.divider()
        st.download_button(
            "Exportar historico (JSON)",
            data=json.dumps(history, ensure_ascii=False, indent=2),
            file_name="orchestration_history.json",
            mime="application/json",
        )
