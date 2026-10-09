"""Solução persistida (cenário + rotas) e sua descrição detalhada.

A descrição detalhada (``route_details``) é a fonte de verdade factual usada pela LLM:
toda instrução, relatório ou resposta deve se apoiar nesses números.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rotas.domain import Scenario, fmt_time
from rotas.evaluation import CostWeights, Evaluation, evaluate_routes, summarize


@dataclass
class Solution:
    scenario: Scenario
    routes: list[list[int]]
    method: str
    meta: dict = field(default_factory=dict)

    def evaluate(self, weights: CostWeights | None = None) -> Evaluation:
        return evaluate_routes(self.scenario, self.routes, weights)

    def save(self, path: Path | str) -> None:
        payload = {
            "method": self.method,
            "meta": self.meta,
            "routes": self.routes,
            "scenario": self.scenario.to_dict(),
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    @classmethod
    def load(cls, path: Path | str) -> Solution:
        d = json.loads(Path(path).read_text())
        return cls(Scenario.from_dict(d["scenario"]), d["routes"], d["method"], d.get("meta", {}))


def route_details(scenario: Scenario, ev: Evaluation) -> list[dict]:
    """Uma entrada por veículo, com cada parada em ordem e horários previstos."""
    out = []
    for j, r in enumerate(ev.routes):
        v = scenario.vehicles[j]
        stops = []
        for order, s in enumerate(r.stops, start=1):
            d = scenario.delivery(s.delivery_id)
            stops.append(
                {
                    "ordem": order,
                    "entrega_id": d.id,
                    "destino": d.location.name,
                    "tipo": d.location.kind,
                    "bairro": d.location.district,
                    "lat": d.location.lat,
                    "lon": d.location.lon,
                    "prioridade": d.priority.label,
                    "refrigerado": d.refrigerated,
                    "carga_kg": d.demand_kg,
                    "itens": list(d.items),
                    "janela": f"{fmt_time(d.window[0])}-{fmt_time(d.window[1])}" if d.window else None,
                    "chegada_prevista": fmt_time(s.arrival_min),
                    "inicio_atendimento": fmt_time(s.start_service_min),
                    "espera_min": round(s.start_service_min - s.arrival_min, 1),
                    "atraso_min": round(s.late_min, 1),
                    "trecho_km": round(s.leg_km, 2),
                }
            )
        out.append(
            {
                "veiculo_id": v.id,
                "veiculo": v.name,
                "tipo": v.kind,
                "refrigerado": v.refrigerated,
                "capacidade_kg": v.capacity_kg,
                "autonomia_km": v.max_range_km,
                "usado": r.used,
                "carga_total_kg": round(r.load_kg, 1),
                "ocupacao_pct": round(100 * r.load_kg / v.capacity_kg, 1),
                "distancia_km": round(r.distance_km, 2),
                "uso_autonomia_pct": round(100 * r.distance_km / v.max_range_km, 1),
                "saida": fmt_time(scenario.shift_start_min),
                "retorno_previsto": fmt_time(r.return_min) if r.used else None,
                "custo_rs": round(r.distance_km * v.cost_per_km, 2),
                "paradas": stops,
            }
        )
    return out


def solution_context(solution: Solution) -> dict:
    """Contexto completo (resumo + rotas) — o que a LLM recebe."""
    ev = solution.evaluate()
    return {
        "cenario": solution.scenario.name,
        "metodo": solution.method,
        "turno": f"{fmt_time(solution.scenario.shift_start_min)}-{fmt_time(solution.scenario.shift_end_min)}",
        "deposito": solution.scenario.depot.name,
        "resumo": summarize(solution.scenario, ev),
        "rotas": route_details(solution.scenario, ev),
    }
