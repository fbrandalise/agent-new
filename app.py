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
    .agent-feedback  { background: linear-gradient(135deg, #ffecd222, #fcb69f22); border-left: 4px solid #f6a623; }
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
        help="Quantas vezes o ciclo avaliacao->feedback->sugestao->execucao roda",
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
        "  |-> Agente 4: Feedback\n"
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
    "Orquestracao de **4 agentes** para avaliacao automatica de qualidade de "
    "enriquecimento de fichas tecnicas de produtos usando **LangGraph** + "
    "**DeepEval**: Avaliador, Feedback Simulado, Sugestor de Prompts e Executor."
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
            "feedback_results": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "history": [],
            "feedback_history": [],
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
        total_steps = max_iterations * 4  # evaluator + feedback + suggester + runner

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

                # --- Feedback ---
                elif node_name == "feedback":
                    status_text.markdown(
                        "**Agente 4 — Feedback Simulado** revisando..."
                    )
                    fb_results = updates.get("feedback_results", [])

                    with col_agents:
                        st.markdown(
                            "<div class='agent-card agent-feedback'>"
                            "<strong>Agente 4 — Feedback Simulado"
                            "</strong></div>",
                            unsafe_allow_html=True,
                        )
                        for fb in fb_results:
                            pos = fb.get("positivos", 0)
                            neg = fb.get("negativos", 0)
                            st.markdown(
                                f"**{fb['prompt_name']}** x "
                                f"**{fb['product_name']}**: "
                                f"<span style='color:#28a745'>"
                                f"+{pos}</span> / "
                                f"<span style='color:#dc3545'>"
                                f"-{neg}</span>",
                                unsafe_allow_html=True,
                            )

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
        st.session_state["feedback_history"] = [
            fb
            for entry in full_history
            for fb in entry.get("feedback", [])
        ]


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

        # ---- Prompt Comparison Visual -----------------------------------------
        st.divider()
        st.subheader("Comparacao Visual dos Prompts")

        # Collect every prompt that appeared across all iterations
        all_prompts_timeline: list[dict] = []
        for entry in history:
            it = entry["iteration"]
            for p in entry.get("prompts_used", []):
                # Average score for this prompt in this iteration
                evals = [
                    e
                    for e in entry.get("evaluations", [])
                    if e["prompt_id"] == p["id"]
                ]
                avg = (
                    sum(e["avg_score"] for e in evals) / len(evals)
                    if evals
                    else None
                )
                # Per-metric averages
                metric_avgs: dict[str, float] = {}
                if evals and evals[0].get("metrics"):
                    for m_name in evals[0]["metrics"]:
                        vals = [
                            e["metrics"][m_name]["score"]
                            for e in evals
                            if m_name in e.get("metrics", {})
                        ]
                        metric_avgs[m_name] = (
                            sum(vals) / len(vals) if vals else 0
                        )
                all_prompts_timeline.append(
                    {
                        "iteration": it,
                        "id": p["id"],
                        "name": p["name"],
                        "rationale": p.get("rationale", ""),
                        "template": p.get("template", ""),
                        "avg_score": avg,
                        "metrics": metric_avgs,
                    }
                )

        if all_prompts_timeline:
            # -- Score evolution line chart --
            fig_line = go.Figure()
            # Group by prompt name
            seen: dict[str, list] = {}
            for pt in all_prompts_timeline:
                seen.setdefault(pt["name"], []).append(pt)
            for pname, entries in seen.items():
                entries.sort(key=lambda x: x["iteration"])
                fig_line.add_trace(
                    go.Scatter(
                        x=[f"Iter {e['iteration']}" for e in entries],
                        y=[e["avg_score"] for e in entries],
                        mode="lines+markers+text",
                        text=[
                            f"{e['avg_score']:.2f}" if e["avg_score"] else ""
                            for e in entries
                        ],
                        textposition="top center",
                        name=pname,
                    )
                )
            fig_line.update_layout(
                title="Evolucao do Score Medio por Prompt",
                yaxis_title="Score Medio",
                xaxis_title="Iteracao",
                yaxis=dict(range=[0, 1]),
                height=400,
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # -- Per-metric grouped bar chart --
            metric_names_set: list[str] = []
            for pt in all_prompts_timeline:
                for m in pt["metrics"]:
                    if m not in metric_names_set:
                        metric_names_set.append(m)

            if metric_names_set:
                fig_metrics = go.Figure()
                for pt in all_prompts_timeline:
                    label = f"Iter {pt['iteration']}: {pt['name']}"
                    fig_metrics.add_trace(
                        go.Bar(
                            name=label,
                            x=metric_names_set,
                            y=[
                                pt["metrics"].get(m, 0)
                                for m in metric_names_set
                            ],
                            text=[
                                f"{pt['metrics'].get(m, 0):.2f}"
                                for m in metric_names_set
                            ],
                            textposition="auto",
                        )
                    )
                fig_metrics.update_layout(
                    title="Scores por Metrica — Todos os Prompts",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    barmode="group",
                    height=450,
                )
                st.plotly_chart(fig_metrics, use_container_width=True)

            # -- Side-by-side prompt cards --
            st.divider()
            st.subheader("Timeline de Prompts")
            for entry in history:
                it = entry["iteration"]
                st.markdown(
                    f"<div class='iteration-header'>"
                    f"Iteracao {it}</div>",
                    unsafe_allow_html=True,
                )
                prompts_in_iter = [
                    pt
                    for pt in all_prompts_timeline
                    if pt["iteration"] == it
                ]
                cols = st.columns(len(prompts_in_iter) or 1)
                for idx, pt in enumerate(prompts_in_iter):
                    with cols[idx]:
                        score = pt["avg_score"]
                        css = _score_class(score) if score is not None else ""
                        score_str = f"{score:.2f}" if score is not None else "—"
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div style='font-size:0.9em;font-weight:bold'>"
                            f"{pt['name']}</div>"
                            f"<div class='{css}' style='margin:4px 0'>"
                            f"{score_str}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(pt["rationale"][:200] if pt["rationale"] else "")
                        with st.expander("Ver template"):
                            st.code(pt["template"], language="text")

                        # Mini metric breakdown
                        if pt["metrics"]:
                            for m_name, m_val in pt["metrics"].items():
                                m_css = _score_class(m_val)
                                st.markdown(
                                    f"<small>{m_name}: "
                                    f"<span class='{m_css}'>{m_val:.2f}"
                                    f"</span></small>",
                                    unsafe_allow_html=True,
                                )

        # ---- Feedback Reinforcement Summary ------------------------------------
        st.divider()
        st.subheader("Feedback do Usuario Simulado")

        # Accumulate feedback across all iterations
        all_feedback: list[dict] = []
        for entry in history:
            it_num = entry["iteration"]
            for fb in entry.get("feedback", []):
                fb_copy = dict(fb)
                fb_copy["iteration"] = it_num
                all_feedback.append(fb_copy)

        if all_feedback:
            # Aggregate totals
            total_pos_all = sum(fb.get("positivos", 0) for fb in all_feedback)
            total_neg_all = sum(fb.get("negativos", 0) for fb in all_feedback)
            total_attrs = total_pos_all + total_neg_all
            approval_rate = (
                total_pos_all / total_attrs * 100 if total_attrs else 0
            )

            # KPI cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Total de Avaliacoes", len(all_feedback))
            with kpi2:
                st.metric("Reforcos Positivos", f"+{total_pos_all}")
            with kpi3:
                st.metric("Reforcos Negativos", f"-{total_neg_all}")
            with kpi4:
                st.metric("Taxa de Aprovacao", f"{approval_rate:.1f}%")

            # Stacked bar: positivos vs negativos per iteration
            iter_pos: dict[int, int] = {}
            iter_neg: dict[int, int] = {}
            for fb in all_feedback:
                it_n = fb["iteration"]
                iter_pos[it_n] = iter_pos.get(it_n, 0) + fb.get("positivos", 0)
                iter_neg[it_n] = iter_neg.get(it_n, 0) + fb.get("negativos", 0)

            iters_sorted = sorted(iter_pos.keys())
            fig_fb = go.Figure()
            fig_fb.add_trace(
                go.Bar(
                    name="Positivos",
                    x=[f"Iter {i}" for i in iters_sorted],
                    y=[iter_pos.get(i, 0) for i in iters_sorted],
                    marker_color="#28a745",
                    text=[f"+{iter_pos.get(i, 0)}" for i in iters_sorted],
                    textposition="auto",
                )
            )
            fig_fb.add_trace(
                go.Bar(
                    name="Negativos",
                    x=[f"Iter {i}" for i in iters_sorted],
                    y=[iter_neg.get(i, 0) for i in iters_sorted],
                    marker_color="#dc3545",
                    text=[f"-{iter_neg.get(i, 0)}" for i in iters_sorted],
                    textposition="auto",
                )
            )
            fig_fb.update_layout(
                title="Reforcos Positivos vs Negativos por Iteracao",
                barmode="group",
                yaxis_title="Quantidade de Atributos",
                height=380,
            )
            st.plotly_chart(fig_fb, use_container_width=True)

            # Approval rate evolution
            approval_by_iter: list[float] = []
            for i in iters_sorted:
                p = iter_pos.get(i, 0)
                n = iter_neg.get(i, 0)
                approval_by_iter.append(
                    p / (p + n) * 100 if (p + n) else 0
                )

            fig_approval = go.Figure()
            fig_approval.add_trace(
                go.Scatter(
                    x=[f"Iter {i}" for i in iters_sorted],
                    y=approval_by_iter,
                    mode="lines+markers+text",
                    text=[f"{v:.0f}%" for v in approval_by_iter],
                    textposition="top center",
                    line=dict(color="#f6a623", width=3),
                    marker=dict(size=10),
                )
            )
            fig_approval.update_layout(
                title="Evolucao da Taxa de Aprovacao",
                yaxis_title="Aprovacao (%)",
                yaxis=dict(range=[0, 105]),
                height=350,
            )
            st.plotly_chart(fig_approval, use_container_width=True)

            # Detailed feedback per iteration
            for entry in history:
                it_n = entry["iteration"]
                fbs = entry.get("feedback", [])
                if not fbs:
                    continue
                with st.expander(
                    f"Feedback detalhado — Iteracao {it_n}"
                ):
                    for fb in fbs:
                        st.markdown(
                            f"**{fb['prompt_name']}** x "
                            f"**{fb['product_name']}** — "
                            f"<span style='color:#28a745'>"
                            f"+{fb.get('positivos', 0)}</span> / "
                            f"<span style='color:#dc3545'>"
                            f"-{fb.get('negativos', 0)}</span>",
                            unsafe_allow_html=True,
                        )
                        if fb.get("comentario_geral"):
                            st.caption(fb["comentario_geral"])
                        items = fb.get("feedbacks", [])
                        if items:
                            for item in items:
                                v = item.get("veredicto", "?")
                                icon = (
                                    "+" if v == "positivo" else "-"
                                )
                                color = (
                                    "#28a745"
                                    if v == "positivo"
                                    else "#dc3545"
                                )
                                st.markdown(
                                    f"<span style='color:{color}'>"
                                    f"[{icon}]</span> "
                                    f"**{item.get('atributo', '?')}** "
                                    f"= {item.get('valor_gerado', '?')} "
                                    f"— {item.get('motivo', '')}",
                                    unsafe_allow_html=True,
                                )
        else:
            st.info("Nenhum feedback disponivel.")

        # Detailed iteration history
        st.divider()
        st.subheader("Historico Detalhado por Iteracao")
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

        # Raw logs
        all_logs = st.session_state.get("all_logs", [])
        if all_logs:
            with st.expander("Logs Completos"):
                st.code("\n".join(all_logs), language="text")

        # ---- Winner Evaluation ------------------------------------------------
        st.divider()
        st.subheader("Avaliacao de Vencedor")

        # Collect best score per prompt across ALL iterations
        prompt_best: dict[str, dict] = {}  # id -> best record
        for pt in all_prompts_timeline:
            score = pt["avg_score"]
            if score is None:
                continue
            pid = pt["id"]
            if pid not in prompt_best or score > prompt_best[pid]["avg_score"]:
                prompt_best[pid] = pt

        if prompt_best:
            ranked = sorted(
                prompt_best.values(),
                key=lambda x: x["avg_score"],
                reverse=True,
            )
            winner = ranked[0]

            # --- Winner banner ---
            st.markdown(
                f"<div style='"
                f"background:linear-gradient(135deg,#28a74522,#20c99722);"
                f"border:2px solid #28a745;border-radius:12px;"
                f"padding:24px;text-align:center;margin-bottom:16px'>"
                f"<div style='font-size:2em'>&#127942;</div>"
                f"<div style='font-size:1.3em;font-weight:bold;"
                f"color:#28a745;margin:8px 0'>"
                f"Vencedor: {winner['name']}</div>"
                f"<div style='font-size:2em;font-weight:bold;"
                f"color:#155724'>{winner['avg_score']:.2f}</div>"
                f"<div style='font-size:0.9em;color:#555;"
                f"margin-top:4px'>Score medio (melhor resultado)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # --- Winner metrics breakdown ---
            if winner.get("metrics"):
                mcols = st.columns(len(winner["metrics"]))
                for i, (m_name, m_val) in enumerate(
                    winner["metrics"].items()
                ):
                    with mcols[i]:
                        css = _score_class(m_val)
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div style='font-size:0.85em'>{m_name}</div>"
                            f"<div class='{css}'>{m_val:.2f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # --- Winner rationale and template ---
            st.markdown(f"**Racional:** {winner.get('rationale', '')}")
            with st.expander("Ver template vencedor"):
                st.code(winner.get("template", ""), language="text")

            # --- Ranking table ---
            if len(ranked) > 1:
                st.markdown("")
                st.markdown("**Ranking completo**")
                for pos, pt in enumerate(ranked, 1):
                    score = pt["avg_score"]
                    css = _score_class(score)
                    medal = (
                        "&#129351;" if pos == 1
                        else "&#129352;" if pos == 2
                        else "&#129353;" if pos == 3
                        else f"#{pos}"
                    )
                    metric_details = "  |  ".join(
                        f"{mn}: {mv:.2f}"
                        for mn, mv in pt.get("metrics", {}).items()
                    )
                    st.markdown(
                        f"<div style='display:flex;align-items:center;"
                        f"gap:12px;padding:8px 12px;"
                        f"border-bottom:1px solid #eee'>"
                        f"<span style='font-size:1.4em;min-width:36px'>"
                        f"{medal}</span>"
                        f"<span style='flex:1;font-weight:600'>"
                        f"{pt['name']}"
                        f"<span style='font-weight:normal;color:#888;"
                        f"font-size:0.85em'> (iter {pt['iteration']})"
                        f"</span></span>"
                        f"<span class='{css}'>{score:.2f}</span>"
                        f"<span style='color:#888;font-size:0.8em'>"
                        f"{metric_details}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # --- Delta vs initial prompts ---
                initial_scores = [
                    pt["avg_score"]
                    for pt in all_prompts_timeline
                    if pt["iteration"] == 1 and pt["avg_score"] is not None
                ]
                if initial_scores:
                    best_initial = max(initial_scores)
                    delta = winner["avg_score"] - best_initial
                    delta_pct = (
                        (delta / best_initial * 100) if best_initial else 0
                    )
                    arrow = "+" if delta >= 0 else ""
                    color = "#28a745" if delta >= 0 else "#dc3545"
                    st.markdown(
                        f"<div style='text-align:center;margin-top:16px;"
                        f"padding:12px;background:#f8f9fa;"
                        f"border-radius:8px'>"
                        f"<span style='font-size:0.95em'>Melhoria vs "
                        f"melhor prompt inicial: </span>"
                        f"<span style='font-size:1.3em;font-weight:bold;"
                        f"color:{color}'>{arrow}{delta:.2f} "
                        f"({arrow}{delta_pct:.1f}%)</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Export
        st.divider()
        st.download_button(
            "Exportar historico (JSON)",
            data=json.dumps(history, ensure_ascii=False, indent=2),
            file_name="orchestration_history.json",
            mime="application/json",
        )
