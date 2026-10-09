"""Simulação de uma semana de operação → histórico para relatórios semanais e sugestões.

Cada dia tem um cenário diferente (demanda e locais variam). O dia é resolvido duas vezes:

* **método anterior** (vizinho mais próximo — como uma roteirização manual/gulosa);
* **algoritmo genético memético** (método proposto).

Todos os indicadores derivados (economias, percentuais, padrões) são calculados aqui, em
Python, para que a LLM apenas os interprete — sem fazer contas.
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from rotas.baselines.heuristics import nearest_neighbor
from rotas.domain import Priority, Scenario
from rotas.evaluation import Evaluation, evaluate_routes, summarize
from rotas.ga.engine import GAConfig, GeneticVRP
from rotas.scenario import generate_scenario

WEEKDAYS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def _driving_hours(scenario: Scenario, ev: Evaluation) -> float:
    return sum(
        r.distance_km / scenario.vehicles[j].speed_kmh for j, r in enumerate(ev.routes) if r.used
    )


def _late_by_district(scenario: Scenario, ev: Evaluation) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for r in ev.routes:
        for s in r.stops:
            if s.late_min > 0:
                out[scenario.delivery(s.delivery_id).location.district] += s.late_min
    return dict(out)


def simulate_day(day_index: int, start: date, generations: int = 800, seed_base: int = 1000) -> dict:
    rng = np.random.default_rng(seed_base + day_index)
    n = int(rng.integers(32, 49))
    sc = generate_scenario(n, seed=seed_base + day_index, name=f"dia_{day_index}")
    ev_nn = evaluate_routes(sc, nearest_neighbor(sc))
    ga = GeneticVRP(sc, GAConfig(generations=generations, local_search=True, seed=day_index)).run()
    ev_ga = ga.best
    s_nn, s_ga = summarize(sc, ev_nn), summarize(sc, ev_ga)
    d = start + timedelta(days=day_index)
    return {
        "data": d.isoformat(),
        "dia_semana": WEEKDAYS[d.weekday()],
        "entregas": n,
        "criticas": sum(x.priority == Priority.CRITICA for x in sc.deliveries),
        "refrigeradas": sum(x.refrigerated for x in sc.deliveries),
        "demanda_kg": round(sum(x.demand_kg for x in sc.deliveries), 1),
        "ag": {
            **{k: s_ga[k] for k in ("distancia_total_km", "custo_operacional_rs", "viavel", "veiculos_usados", "tempo_medio_ate_criticas_h")},
            "horas_conducao": round(_driving_hours(sc, ev_ga), 2),
            "violacoes": s_ga["violacoes"],
            "atraso_por_bairro_min": _late_by_district(sc, ev_ga),
            "ocupacao_por_veiculo_pct": {
                sc.vehicles[j].name: round(100 * r.load_kg / sc.vehicles[j].capacity_kg, 1) for j, r in enumerate(ev_ga.routes)
            },
            "km_por_veiculo": {sc.vehicles[j].name: round(r.distance_km, 1) for j, r in enumerate(ev_ga.routes)},
        },
        "anterior": {
            **{k: s_nn[k] for k in ("distancia_total_km", "custo_operacional_rs", "viavel", "veiculos_usados", "tempo_medio_ate_criticas_h")},
            "horas_conducao": round(_driving_hours(sc, ev_nn), 2),
            "violacoes": s_nn["violacoes"],
        },
    }


def _pct(a: float, b: float) -> float:
    return round(100 * (b - a) / b, 1) if b else 0.0


def aggregate(days: list[dict]) -> dict:
    km_ag = sum(d["ag"]["distancia_total_km"] for d in days)
    km_nn = sum(d["anterior"]["distancia_total_km"] for d in days)
    rs_ag = sum(d["ag"]["custo_operacional_rs"] for d in days)
    rs_nn = sum(d["anterior"]["custo_operacional_rs"] for d in days)
    h_ag = sum(d["ag"]["horas_conducao"] for d in days)
    h_nn = sum(d["anterior"]["horas_conducao"] for d in days)
    for d in days:
        d["economia_km"] = round(d["anterior"]["distancia_total_km"] - d["ag"]["distancia_total_km"], 1)
        d["economia_pct"] = _pct(d["ag"]["distancia_total_km"], d["anterior"]["distancia_total_km"])

    occupancy: dict[str, list[float]] = defaultdict(list)
    km_vehicle: dict[str, float] = defaultdict(float)
    late_district: dict[str, float] = defaultdict(float)
    for d in days:
        for v, o in d["ag"]["ocupacao_por_veiculo_pct"].items():
            occupancy[v].append(o)
        for v, k in d["ag"]["km_por_veiculo"].items():
            km_vehicle[v] += k
        for b, m in d["ag"]["atraso_por_bairro_min"].items():
            late_district[b] += m

    crit_ag = [d["ag"]["tempo_medio_ate_criticas_h"] for d in days if d["ag"]["tempo_medio_ate_criticas_h"] is not None]
    crit_nn = [d["anterior"]["tempo_medio_ate_criticas_h"] for d in days if d["anterior"]["tempo_medio_ate_criticas_h"] is not None]
    return {
        "periodo": f"{days[0]['data']} a {days[-1]['data']}",
        "totais": {
            "dias": len(days),
            "entregas": sum(d["entregas"] for d in days),
            "entregas_criticas": sum(d["criticas"] for d in days),
            "entregas_refrigeradas": sum(d["refrigeradas"] for d in days),
            "km_otimizado": round(km_ag, 1),
            "km_metodo_anterior": round(km_nn, 1),
            "economia_km": round(km_nn - km_ag, 1),
            "economia_km_pct": _pct(km_ag, km_nn),
            "custo_otimizado_rs": round(rs_ag, 2),
            "custo_metodo_anterior_rs": round(rs_nn, 2),
            "economia_rs": round(rs_nn - rs_ag, 2),
            "horas_conducao_otimizado": round(h_ag, 1),
            "horas_conducao_anterior": round(h_nn, 1),
            "economia_horas": round(h_nn - h_ag, 1),
            "dias_viaveis_otimizado": sum(d["ag"]["viavel"] for d in days),
            "dias_viaveis_anterior": sum(d["anterior"]["viavel"] for d in days),
            "tempo_medio_ate_criticas_h_otimizado": round(float(np.mean(crit_ag)), 2) if crit_ag else None,
            "tempo_medio_ate_criticas_h_anterior": round(float(np.mean(crit_nn)), 2) if crit_nn else None,
        },
        "padroes": {
            "ocupacao_media_por_veiculo_pct": {v: round(float(np.mean(o)), 1) for v, o in occupancy.items()},
            "dias_sem_uso_por_veiculo": {v: sum(x == 0 for x in o) for v, o in occupancy.items()},
            "km_semana_por_veiculo": {v: round(k, 1) for v, k in km_vehicle.items()},
            "atraso_total_por_bairro_min": dict(sorted(late_district.items(), key=lambda x: -x[1])),
            "dia_maior_demanda": max(days, key=lambda d: d["demanda_kg"])["dia_semana"],
            "dia_menor_economia": min(days, key=lambda d: d["economia_pct"])["dia_semana"],
        },
        "dias": days,
    }


def simulate_week(n_days: int = 7, start: date | None = None, generations: int = 800) -> dict:
    start = start or date(2026, 10, 5)
    with ProcessPoolExecutor() as ex:
        days = list(ex.map(simulate_day, range(n_days), [start] * n_days, [generations] * n_days))
    return aggregate(days)


def save_week(history: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(history, indent=2, ensure_ascii=False, default=str))
