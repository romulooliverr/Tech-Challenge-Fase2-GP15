"""Decodificação e avaliação de soluções (usado pelo AG E por todos os baselines).

Representação: permutação de ``n + K - 1`` genes inteiros distintos:

* ``1..n``         → entregas;
* ``n+1..n+K-1``   → separadores de rota.

Os separadores cortam a permutação em K segmentos; o segmento j é a rota do veículo j.
Exemplo (n=6, K=3): ``[3, 1, 7, 5, 2, 8, 6, 4]`` → v1: 3→1 | v2: 5→2 | v3: 6→4.

Toda solução — inclusive as dos baselines — é avaliada pela MESMA função, o que torna a
comparação entre métodos justa. Restrições violadas não descartam a solução: geram
penalidades proporcionais à violação (restrições "suaves"), o que permite ao AG atravessar
regiões inviáveis do espaço de busca rumo a soluções viáveis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from rotas.domain import Priority, Scenario


@dataclass(frozen=True)
class CostWeights:
    # peso (km-equivalente por hora de espera) de cada prioridade: quanto "custa" um
    # medicamento crítico ficar 1 h a mais na estrada
    priority_km_per_hour: dict[int, float] = field(
        default_factory=lambda: {Priority.CRITICA: 6.0, Priority.URGENTE: 2.0, Priority.REGULAR: 0.0}
    )
    vehicle_fixed_km: float = 5.0  # custo fixo por veículo acionado
    overload_per_kg: float = 10.0
    over_range_per_km: float = 10.0
    refrigeration_per_delivery: float = 300.0  # item de cadeia fria em veículo sem refrigeração
    late_per_min: float = 10.0  # atraso em janela de atendimento domiciliar
    overtime_per_min: float = 10.0  # retorno ao depósito após o fim do turno

    def to_dict(self) -> dict:
        d = asdict(self)
        d["priority_km_per_hour"] = {int(k): v for k, v in self.priority_km_per_hour.items()}
        return d


@dataclass
class Stop:
    delivery_id: int
    arrival_min: float
    start_service_min: float  # = chegada, ou início da janela se chegou cedo
    departure_min: float
    leg_km: float
    load_after_kg: float
    late_min: float


@dataclass
class RouteResult:
    vehicle_id: int
    sequence: list[int]
    stops: list[Stop]
    distance_km: float
    load_kg: float
    return_min: float
    overload_kg: float
    over_range_km: float
    refrigeration_violations: int
    late_min: float
    overtime_min: float

    @property
    def used(self) -> bool:
        return bool(self.sequence)


@dataclass
class Evaluation:
    cost: float
    distance_km: float
    priority_cost: float
    penalty: float
    routes: list[RouteResult]

    @property
    def feasible(self) -> bool:
        return self.penalty < 1e-9

    @property
    def vehicles_used(self) -> int:
        return sum(r.used for r in self.routes)

    def violations(self) -> dict[str, float]:
        return {
            "overload_kg": sum(r.overload_kg for r in self.routes),
            "over_range_km": sum(r.over_range_km for r in self.routes),
            "refrigeration": sum(r.refrigeration_violations for r in self.routes),
            "late_min": sum(r.late_min for r in self.routes),
            "overtime_min": sum(r.overtime_min for r in self.routes),
        }


def split_routes(chromosome: Sequence[int], n_deliveries: int, n_vehicles: int) -> list[list[int]]:
    routes: list[list[int]] = [[]]
    for g in chromosome:
        if g > n_deliveries:
            routes.append([])
        else:
            routes[-1].append(int(g))
    if len(routes) != n_vehicles:
        raise ValueError(f"Cromossomo gerou {len(routes)} rotas para {n_vehicles} veículos.")
    return routes


def join_routes(routes: Sequence[Sequence[int]], n_deliveries: int) -> list[int]:
    """Inverso de ``split_routes``: rotas por veículo → cromossomo com separadores."""
    chrom: list[int] = []
    for j, r in enumerate(routes):
        if j > 0:
            chrom.append(n_deliveries + j)
        chrom.extend(int(x) for x in r)
    return chrom


def simulate_route(scenario: Scenario, vehicle_idx: int, sequence: Sequence[int]) -> RouteResult:
    v = scenario.vehicles[vehicle_idx]
    dist = scenario.dist
    t = float(scenario.shift_start_min)
    prev, km, load = 0, 0.0, 0.0
    stops: list[Stop] = []
    refrig = 0
    late = 0.0
    for did in sequence:
        d = scenario.deliveries[did - 1]
        leg = float(dist[prev, did])
        km += leg
        t += leg / v.speed_kmh * 60.0
        arrival = t
        start = arrival
        late_here = 0.0
        if d.window:
            start = max(arrival, d.window[0])  # chegou cedo: espera abrir a janela
            late_here = max(0.0, arrival - d.window[1])
        t = start + d.service_min
        load += d.demand_kg
        if d.refrigerated and not v.refrigerated:
            refrig += 1
        late += late_here
        stops.append(Stop(did, arrival, start, t, leg, load, late_here))
        prev = did
    if sequence:
        back = float(dist[prev, 0])
        km += back
        t += back / v.speed_kmh * 60.0
    return RouteResult(
        vehicle_id=v.id,
        sequence=list(sequence),
        stops=stops,
        distance_km=km,
        load_kg=load,
        return_min=t,
        overload_kg=max(0.0, load - v.capacity_kg),
        over_range_km=max(0.0, km - v.max_range_km),
        refrigeration_violations=refrig,
        late_min=late,
        overtime_min=max(0.0, t - scenario.shift_end_min) if sequence else 0.0,
    )


def _route_terms(scenario: Scenario, r: RouteResult, w: CostWeights) -> tuple[float, float, float]:
    """(custo de prioridade, penalidade, custo fixo) de uma rota — o custo é separável por rota."""
    priority_cost = 0.0
    for s in r.stops:
        p = scenario.deliveries[s.delivery_id - 1].priority
        hours = (s.start_service_min - scenario.shift_start_min) / 60.0
        priority_cost += w.priority_km_per_hour.get(p, 0.0) * hours
    penalty = (
        w.overload_per_kg * r.overload_kg
        + w.over_range_per_km * r.over_range_km
        + w.refrigeration_per_delivery * r.refrigeration_violations
        + w.late_per_min * r.late_min
        + w.overtime_per_min * r.overtime_min
    )
    fixed = w.vehicle_fixed_km if r.used else 0.0
    return priority_cost, penalty, fixed


def route_cost(
    scenario: Scenario, vehicle_idx: int, sequence: Sequence[int], weights: CostWeights | None = None
) -> float:
    """Custo de uma única rota (usado pela busca local 2-opt sem reavaliar a solução toda)."""
    r = simulate_route(scenario, vehicle_idx, sequence)
    return r.distance_km + sum(_route_terms(scenario, r, weights or CostWeights()))


def evaluate_routes(
    scenario: Scenario, routes: Sequence[Sequence[int]], weights: CostWeights | None = None
) -> Evaluation:
    w = weights or CostWeights()
    results = [simulate_route(scenario, j, r) for j, r in enumerate(routes)]
    terms = [_route_terms(scenario, r, w) for r in results]
    distance = sum(r.distance_km for r in results)
    priority_cost = sum(t[0] for t in terms)
    penalty = sum(t[1] for t in terms)
    fixed = sum(t[2] for t in terms)
    return Evaluation(
        cost=distance + priority_cost + penalty + fixed,
        distance_km=distance,
        priority_cost=priority_cost,
        penalty=penalty,
        routes=results,
    )


def evaluate_chromosome(
    scenario: Scenario, chromosome: Sequence[int], weights: CostWeights | None = None
) -> Evaluation:
    routes = split_routes(chromosome, scenario.n_deliveries, scenario.n_vehicles)
    return evaluate_routes(scenario, routes, weights)


def summarize(scenario: Scenario, ev: Evaluation) -> dict:
    """Resumo numérico da solução — também é o contexto factual entregue à LLM."""
    critical_hours = [
        (s.start_service_min - scenario.shift_start_min) / 60
        for r in ev.routes
        for s in r.stops
        if scenario.deliveries[s.delivery_id - 1].priority == Priority.CRITICA
    ]
    cost_rs = sum(
        r.distance_km * scenario.vehicles[j].cost_per_km for j, r in enumerate(ev.routes)
    )
    return {
        "custo_objetivo": round(ev.cost, 2),
        "distancia_total_km": round(ev.distance_km, 2),
        "custo_operacional_rs": round(cost_rs, 2),
        "viavel": ev.feasible,
        "penalidade": round(ev.penalty, 2),
        "veiculos_usados": ev.vehicles_used,
        "entregas": scenario.n_deliveries,
        "tempo_medio_ate_criticas_h": round(sum(critical_hours) / len(critical_hours), 2)
        if critical_hours
        else None,
        "tempo_max_ate_criticas_h": round(max(critical_hours), 2) if critical_hours else None,
        "violacoes": {k: round(v, 2) for k, v in ev.violations().items()},
    }
