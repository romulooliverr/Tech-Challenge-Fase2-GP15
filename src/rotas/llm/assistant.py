"""Assistente de rotas: instruções a motoristas, relatórios, sugestões e chat com ferramentas."""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Callable

import pandas as pd

from rotas.llm import prompts
from rotas.llm.provider import ChatSession, LLMProvider, get_provider
from rotas.solution import Solution, solution_context


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def comparison_indicators(comparison: pd.DataFrame | None, ga_method: str) -> dict:
    """Percentuais pré-calculados do AG frente a cada método (a LLM não faz contas)."""
    if comparison is None or comparison.empty:
        return {}
    ga = comparison[comparison["metodo"] == ga_method]
    if ga.empty:
        ga = comparison[comparison["metodo"].str.startswith("AG")].head(1)
    g = ga.iloc[0]
    out = {}
    for _, row in comparison.iterrows():
        if row["metodo"] == g["metodo"]:
            continue
        out[row["metodo"]] = {
            "distancia_km": row["distancia_total_km"],
            "viavel": bool(row["viavel"]),
            "diferenca_distancia_ag_km": round(row["distancia_total_km"] - g["distancia_total_km"], 1),
            "economia_distancia_ag_pct": round(100 * (row["distancia_total_km"] - g["distancia_total_km"]) / row["distancia_total_km"], 1),
            "economia_custo_objetivo_ag_pct": round(100 * (row["custo_objetivo"] - g["custo_objetivo"]) / row["custo_objetivo"], 1),
        }
    return out


def context_as_tables(ctx: dict, comparison: dict | None = None) -> str:
    """Contexto do chat em tabelas Markdown compactas (uma linha por parada).

    Achado da avaliação: com JSON indentado e longo, o modelo "lite" deixou de encontrar
    entregas críticas de um veículo; tabelas + índice por prioridade resolveram.
    """
    r = ctx["resumo"]
    lines = [
        f"Cenário: {ctx['cenario']} | Método: {ctx['metodo']} | Turno: {ctx['turno']} | Depósito: {ctx['deposito']}",
        "Resumo: " + "; ".join(f"{k}={v}" for k, v in r.items()),
        "",
        "ÍNDICE — entregas críticas e refrigeradas por veículo:",
    ]
    for rt in ctx["rotas"]:
        crit = [f"#{p['ordem']} {p['destino']} ({p['chegada_prevista']})" for p in rt["paradas"] if p["prioridade"] == "crítica"]
        refr = [f"#{p['ordem']} {p['destino']}" for p in rt["paradas"] if p["refrigerado"]]
        lines.append(f"- {rt['veiculo']}: {len(crit)} críticas [{'; '.join(crit) or 'nenhuma'}] | {len(refr)} refrigeradas [{'; '.join(refr) or 'nenhuma'}]")
    for rt in ctx["rotas"]:
        lines += [
            "",
            f"### {rt['veiculo']} (id {rt['veiculo_id']}, {rt['tipo']}, refrigerado={rt['refrigerado']}) — "
            f"{len(rt['paradas'])} paradas, {rt['distancia_km']} km, carga {rt['carga_total_kg']} kg "
            f"({rt['ocupacao_pct']}% da capacidade), autonomia usada {rt['uso_autonomia_pct']}%, "
            f"saída {rt['saida']}, retorno {rt['retorno_previsto'] or '—'}, custo R$ {rt['custo_rs']}",
        ]
        if rt["paradas"]:
            lines.append("| ordem | destino | bairro | prioridade | refrigerado | chegada | janela | carga kg | itens |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for p in rt["paradas"]:
                lines.append(
                    f"| {p['ordem']} | {p['destino']} | {p['bairro']} | {p['prioridade']} | "
                    f"{'sim' if p['refrigerado'] else 'não'} | {p['chegada_prevista']} | {p['janela'] or '—'} | "
                    f"{p['carga_kg']} | {', '.join(p['itens'])} |"
                )
    if comparison:
        lines += ["", "Comparativo do AG com outros métodos (valores já calculados):"]
        for m, v in comparison.items():
            lines.append(f"- {m}: " + "; ".join(f"{k}={x}" for k, x in v.items()))
    return "\n".join(lines)


class RouteAssistant:
    def __init__(
        self,
        solution: Solution,
        provider: LLMProvider | None = None,
        comparison: pd.DataFrame | None = None,
        weekly_history: dict | None = None,
    ) -> None:
        self.solution = solution
        self.provider = provider or get_provider()
        self.comparison = comparison
        self.weekly_history = weekly_history
        self.context = solution_context(solution)

    # ------------------------------------------------------------------ instruções
    def route_data(self, vehicle_id: int) -> dict:
        for r in self.context["rotas"]:
            if r["veiculo_id"] == vehicle_id:
                return {k: v for k, v in r.items() if k != "usado"}
        raise ValueError(f"Veículo {vehicle_id} não existe.")

    def driver_instructions(self, vehicle_id: int, prompt_version: str = "v1") -> str:
        data = self.route_data(vehicle_id)
        if not data["paradas"]:
            return f"{data['veiculo']}: sem entregas hoje."
        template = prompts.DRIVER_V1 if prompt_version == "v1" else prompts.DRIVER_V0
        system = prompts.SYSTEM_BASE if prompt_version == "v1" else ""
        return self.provider.generate(system, prompts.render(template, data))

    # ------------------------------------------------------------------ relatórios
    def daily_report(self) -> str:
        data = {
            "metodo": self.context["metodo"],
            "turno": self.context["turno"],
            "resumo": self.context["resumo"],
            "frota": [
                {k: r[k] for k in ("veiculo", "usado", "carga_total_kg", "ocupacao_pct", "distancia_km", "uso_autonomia_pct", "retorno_previsto", "custo_rs")}
                | {"paradas": len(r["paradas"]),
                   "criticas": sum(p["prioridade"] == "crítica" for p in r["paradas"]),
                   "refrigeradas": sum(p["refrigerado"] for p in r["paradas"])}
                for r in self.context["rotas"]
            ],
            "indicadores": {"comparativo": comparison_indicators(self.comparison, self.solution.method)},
        }
        return self.provider.generate(prompts.SYSTEM_BASE, prompts.render(prompts.DAILY_REPORT, data))

    def _weekly(self) -> dict:
        if not self.weekly_history:
            raise ValueError("Histórico semanal não carregado (rode `rotas simulate-week`).")
        h = self.weekly_history
        days = [
            {k: d[k] for k in ("data", "dia_semana", "entregas", "criticas", "refrigeradas", "economia_km", "economia_pct")}
            | {"km_otimizado": d["ag"]["distancia_total_km"], "km_anterior": d["anterior"]["distancia_total_km"],
               "tempo_medio_criticas_h_otimizado": d["ag"]["tempo_medio_ate_criticas_h"],
               "viavel_otimizado": d["ag"]["viavel"], "viavel_anterior": d["anterior"]["viavel"]}
            for d in h["dias"]
        ]
        return {"periodo": h["periodo"], "totais": h["totais"], "padroes": h["padroes"], "dias": days}

    def weekly_report(self) -> str:
        return self.provider.generate(prompts.SYSTEM_BASE, prompts.render(prompts.WEEKLY_REPORT, self._weekly()))

    def improvement_suggestions(self) -> str:
        return self.provider.generate(prompts.SYSTEM_BASE, prompts.render(prompts.SUGGESTIONS, self._weekly()), temperature=0.5)

    # ------------------------------------------------------------------ chat com ferramentas
    def tools(self) -> list[Callable]:
        ctx = self.context

        def metricas_gerais() -> dict:
            """Retorna o resumo da solução do dia: distância total, custo, viabilidade,
            veículos usados e tempo médio até as entregas críticas."""
            return {"metodo": ctx["metodo"], "turno": ctx["turno"], **ctx["resumo"]}

        def listar_veiculos() -> list[dict]:
            """Lista os veículos com distância, carga, ocupação, número de paradas e horário de retorno."""
            return [
                {k: r[k] for k in ("veiculo_id", "veiculo", "tipo", "refrigerado", "usado", "distancia_km", "carga_total_kg", "ocupacao_pct", "retorno_previsto")}
                | {"paradas": len(r["paradas"])}
                for r in ctx["rotas"]
            ]

        def rota_do_veiculo(veiculo_id: int) -> dict:
            """Retorna a rota completa (paradas em ordem, horários, itens) de um veículo pelo seu ID numérico."""
            for r in ctx["rotas"]:
                if r["veiculo_id"] == veiculo_id:
                    return r
            return {"erro": f"veículo {veiculo_id} não encontrado"}

        def buscar_entrega(termo: str) -> dict:
            """Busca entregas cujo destino, bairro, tipo de unidade ou item contenha o termo
            (sem diferenciar acentos/maiúsculas). Também aceita o número da entrega."""
            t = _norm(termo.strip())
            found = []
            for r in ctx["rotas"]:
                for p in r["paradas"]:
                    hay = _norm(" ".join([p["destino"], p["bairro"], p["tipo"], " ".join(p["itens"])]))
                    if t == str(p["entrega_id"]) or (t and t in hay):
                        found.append({"veiculo": r["veiculo"], **p})
            return {"termo": termo, "resultados": found[:15]}

        def entregas_por_prioridade(prioridade: str) -> dict:
            """Lista as entregas de uma prioridade: 'crítica', 'urgente' ou 'regular'."""
            alvo = _norm(prioridade)
            found = [
                {"veiculo": r["veiculo"], "ordem": p["ordem"], "destino": p["destino"], "chegada_prevista": p["chegada_prevista"], "refrigerado": p["refrigerado"]}
                for r in ctx["rotas"] for p in r["paradas"] if _norm(p["prioridade"]) == alvo
            ]
            return {"prioridade": prioridade, "total": len(found), "entregas": found}

        def comparar_metodos() -> dict:
            """Compara o algoritmo genético com os outros métodos de roteamento (percentuais já calculados)."""
            return comparison_indicators(self.comparison, self.solution.method) or {"erro": "comparativo indisponível"}

        return [metricas_gerais, listar_veiculos, rota_do_veiculo, buscar_entrega, entregas_por_prioridade, comparar_metodos]

    def start_chat(self, mode: str | None = None) -> ChatSession:
        """Dois modos de chat (``CHAT_MODE``):

        * ``context`` (padrão) — todos os dados do dia vão no system prompt: 1 requisição por
          pergunta. Viável aqui porque o contexto é pequeno (~5 mil tokens) e econômico no
          plano gratuito do Gemini.
        * ``tools`` — *function calling*: o modelo chama funções Python (buscar_entrega,
          rota_do_veiculo...) para obter os fatos. Escala para bases grandes, mas cada chamada
          de ferramenta consome uma requisição extra.
        """
        mode = (mode or os.getenv("CHAT_MODE", "context")).lower()
        if mode == "tools":
            return self.provider.start_chat(prompts.CHAT_SYSTEM, self.tools())
        tables = context_as_tables(self.context, comparison_indicators(self.comparison, self.solution.method))
        return self.provider.start_chat(prompts.CHAT_SYSTEM_CONTEXT.format(dados=tables), [])
