"""Interface de demonstração: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from rotas.domain import fmt_time
from rotas.evaluation import summarize
from rotas.ga.engine import GAConfig, GeneticVRP
from rotas.llm.assistant import RouteAssistant
from rotas.llm.provider import get_provider
from rotas.scenario import generate_scenario
from rotas.solution import Solution, route_details
from rotas.viz.maps import route_map

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

st.set_page_config(page_title="Rotas Hospitalares — AG + LLM", page_icon="🚑", layout="wide")


@st.cache_resource
def provider():
    return get_provider()


def load_csv(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def llm(label: str, fn, *args) -> str | None:
    """Chama a LLM com indicador de carregamento visível (e cronômetro) e erro amigável."""
    try:
        with st.spinner(f"⏳ {label} com **{prov.name}**... (modelos locais podem levar até 1 min)", show_time=True):
            out = fn(*args)
    except Exception as e:  # noqa: BLE001 — qualquer falha da LLM vira mensagem na tela, não traceback
        st.error(f"Não foi possível gerar o texto: {e}")
        return None
    st.toast(f"{label}: pronto!", icon="✅")
    return out


def load_week() -> dict | None:
    p = RESULTS / "weekly_history.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ barra lateral
st.sidebar.title("🚑 Rotas Hospitalares")
st.sidebar.caption("Otimização de rotas de medicamentos e insumos — São Paulo")
source = st.sidebar.radio("Solução", ["Otimizar agora", "Melhor solução dos experimentos"])

if source == "Otimizar agora":
    with st.sidebar.form("ga"):
        st.subheader("Cenário")
        n = st.slider("Entregas", 10, 80, 40, 5)
        scen_seed = st.number_input("Seed do cenário", 0, 9999, 42)
        st.subheader("Algoritmo genético")
        pop = st.slider("População", 20, 300, 150, 10)
        gens = st.slider("Gerações", 50, 3000, 1000, 50)
        mut = st.slider("Taxa de mutação", 0.0, 1.0, 0.6, 0.05)
        cx = st.selectbox("Cruzamento", ["ox", "pmx", "erx"])
        memetic = st.checkbox("Memético (relocate + 2-opt)", value=True)
        run = st.form_submit_button("▶ Otimizar", width="stretch")
    if run:
        sc = generate_scenario(n, int(scen_seed))
        progress = st.sidebar.progress(0.0, "Evoluindo...")
        chart = st.sidebar.empty()
        hist: list[dict] = []

        def on_gen(stats, _best):
            hist.append({"geração": stats.generation, "melhor custo": stats.best_cost})
            if stats.generation % 25 == 0 or stats.generation == gens:
                progress.progress(stats.generation / gens, f"Geração {stats.generation}/{gens}")
                chart.line_chart(pd.DataFrame(hist).set_index("geração"), height=160)

        res = GeneticVRP(sc, GAConfig(population_size=pop, generations=gens, mutation_rate=mut,
                                      crossover=cx, local_search=memetic), callbacks=[on_gen]).run()
        st.session_state.solution = Solution(sc, res.routes, f"AG ({'memético' if memetic else 'padrão'}, {cx.upper()})")
        st.session_state.history = hist
        st.session_state.pop("chat", None)
        st.session_state.llm_cache = {}
elif (RESULTS / "best_solution.json").exists():
    if st.session_state.get("source") != source:
        st.session_state.solution = Solution.load(RESULTS / "best_solution.json")
        st.session_state.history = None
        st.session_state.pop("chat", None)
        st.session_state.llm_cache = {}
else:
    st.sidebar.warning("Rode `rotas experiments` para gerar a melhor solução.")
st.session_state.source = source

prov = provider()
st.sidebar.divider()
st.sidebar.caption(f"LLM: **{prov.name}**" + (f" · {getattr(prov, 'model', '')}" if prov.name in ("gemini", "ollama") else " (offline)"))

sol: Solution | None = st.session_state.get("solution")
if sol is None:
    st.info("Configure o cenário e clique em **Otimizar** na barra lateral.")
    st.stop()

ev = sol.evaluate()
summary = summarize(sol.scenario, ev)
details = route_details(sol.scenario, ev)
comparison = load_csv("methods_comparison.csv")
assistant = RouteAssistant(sol, provider=prov, comparison=comparison, weekly_history=load_week())
cache: dict = st.session_state.setdefault("llm_cache", {})

st.title("Distribuição de medicamentos e insumos")
st.caption(f"{sol.method} · cenário {sol.scenario.name} · turno {fmt_time(sol.scenario.shift_start_min)}–{fmt_time(sol.scenario.shift_end_min)}")
c = st.columns(6)
c[0].metric("Distância", f"{summary['distancia_total_km']} km")
c[1].metric("Custo operacional", f"R$ {summary['custo_operacional_rs']:.2f}")
c[2].metric("Veículos", summary["veiculos_usados"])
c[3].metric("Entregas", summary["entregas"])
c[4].metric("Até críticas (média)", f"{summary['tempo_medio_ate_criticas_h']} h")
c[5].metric("Restrições", "✅ atendidas" if summary["viavel"] else "⚠️ violadas")

tab_map, tab_cmp, tab_instr, tab_rep, tab_chat = st.tabs(
    ["🗺️ Mapa e rotas", "📊 Comparativo", "🚚 Instruções", "📈 Relatórios", "💬 Pergunte às rotas"]
)

with tab_map:
    st_folium(route_map(sol), height=560, use_container_width=True, returned_objects=[])
    fleet = pd.DataFrame([
        {"Veículo": r["veiculo"], "Paradas": len(r["paradas"]), "Distância (km)": r["distancia_km"],
         "Carga (kg)": r["carga_total_kg"], "Ocupação %": r["ocupacao_pct"], "Autonomia usada %": r["uso_autonomia_pct"],
         "Retorno": r["retorno_previsto"] or "—",
         "Críticas": sum(p["prioridade"] == "crítica" for p in r["paradas"]),
         "Refrigeradas": sum(p["refrigerado"] for p in r["paradas"])}
        for r in details
    ])
    st.dataframe(fleet, hide_index=True, width="stretch")
    if not summary["viavel"]:
        st.warning(f"Violações: {summary['violacoes']}")
    if st.session_state.get("history"):
        st.subheader("Convergência")
        st.line_chart(pd.DataFrame(st.session_state.history).set_index("geração"))

with tab_cmp:
    if comparison is None:
        st.info("Rode `rotas experiments` para gerar o comparativo.")
    else:
        st.subheader("AG × outras abordagens de roteamento (mesmo cenário, mesma função de custo)")
        st.dataframe(comparison[["metodo", "custo_objetivo", "distancia_total_km", "custo_operacional_rs", "viavel",
                                 "tempo_medio_ate_criticas_h", "veiculos_usados", "tempo_s"]],
                     hide_index=True, width="stretch")
        for img in ["methods_comparison.png", "convergence.png", "boxplot_cost.png"]:
            if (RESULTS / img).exists():
                st.image(str(RESULTS / img))
        summ = load_csv("experiments_summary.csv")
        if summ is not None:
            st.subheader("Experimentos do AG (média de 5 seeds)")
            st.dataframe(summ, hide_index=True, width="stretch")

with tab_instr:
    used = [r for r in details if r["usado"]]
    choice = st.selectbox("Veículo", used, format_func=lambda r: f"{r['veiculo']} — {len(r['paradas'])} paradas")
    if st.button("Gerar instruções para o motorista", type="primary"):
        out = llm("Gerando instruções para o motorista", assistant.driver_instructions, choice["veiculo_id"])
        if out is not None:
            cache[("instr", choice["veiculo_id"])] = out
    if ("instr", choice["veiculo_id"]) in cache:
        st.markdown(cache[("instr", choice["veiculo_id"])])

with tab_rep:
    kind = st.radio("Tipo", ["Diário", "Semanal", "Sugestões de melhoria"], horizontal=True)
    week = load_week()
    if kind != "Diário" and week:
        t = week["totais"]
        k = st.columns(4)
        k[0].metric("Economia de km na semana", f"{t['economia_km']} km", f"{t['economia_km_pct']}%")
        k[1].metric("Economia (R$)", f"R$ {t['economia_rs']:.2f}")
        k[2].metric("Horas de condução poupadas", f"{t['economia_horas']} h")
        k[3].metric("Até críticas (média)", f"{t['tempo_medio_ate_criticas_h_otimizado']} h",
                    f"{t['tempo_medio_ate_criticas_h_otimizado'] - t['tempo_medio_ate_criticas_h_anterior']:.2f} h", delta_color="inverse")
    if kind != "Diário" and not week:
        st.info("Rode `rotas simulate-week` para gerar o histórico semanal.")
    elif st.button(f"Gerar: {kind}", type="primary"):
        fn = {"Diário": assistant.daily_report, "Semanal": assistant.weekly_report,
              "Sugestões de melhoria": assistant.improvement_suggestions}[kind]
        out = llm(f"Gerando {kind.lower()}", fn)
        if out is not None:
            cache[("rep", kind)] = out
    if ("rep", kind) in cache:
        st.markdown(cache[("rep", kind)])

with tab_chat:
    st.caption("Ex.: *Quais entregas críticas a Van Refrigerada 2 faz?* · *Quem leva insulina para o Tatuapé?* · "
               "*Quanto economizamos em relação ao vizinho mais próximo?*")
    if "chat" not in st.session_state:
        st.session_state.chat = assistant.start_chat()
        st.session_state.messages = []
    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])
    if q := st.chat_input("Pergunte sobre as rotas e entregas do dia"):
        st.chat_message("user").markdown(q)
        with st.chat_message("assistant"):
            a = llm("Consultando as rotas", st.session_state.chat.send, q)
            if a is not None:
                st.markdown(a)
        if a is not None:
            st.session_state.messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
