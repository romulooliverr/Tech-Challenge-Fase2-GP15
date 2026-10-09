"""Heurísticas clássicas de roteamento usadas como baseline (e para semear o AG)."""

from __future__ import annotations

import numpy as np

from rotas.domain import Scenario
from rotas.evaluation import CostWeights, join_routes, route_cost, simulate_route

Routes = list[list[int]]


def cheapest_insertion(
    scenario: Scenario, routes: Routes, delivery_id: int, weights: CostWeights
) -> None:
    """Insere a entrega na posição (veículo, índice) de menor aumento de custo. Altera ``routes``."""
    best = (np.inf, 0, 0)
    for j, r in enumerate(routes):
        base = route_cost(scenario, j, r, weights)
        for pos in range(len(r) + 1):
            delta = route_cost(scenario, j, r[:pos] + [delivery_id] + r[pos:], weights) - base
            if delta < best[0]:
                best = (delta, j, pos)
    _, j, pos = best
    routes[j].insert(pos, delivery_id)


def random_solution(scenario: Scenario, rng: np.random.Generator) -> list[int]:
    n, k = scenario.n_deliveries, scenario.n_vehicles
    return [int(g) for g in rng.permutation(np.arange(1, n + k))]


def random_search(
    scenario: Scenario, n_samples: int = 2000, seed: int = 0, weights: CostWeights | None = None
) -> Routes:
    """Melhor de N soluções aleatórias — referência de "pior caso razoável"."""
    from rotas.evaluation import evaluate_chromosome, split_routes

    rng = np.random.default_rng(seed)
    best = min(
        (random_solution(scenario, rng) for _ in range(n_samples)),
        key=lambda c: evaluate_chromosome(scenario, c, weights).cost,
    )
    return split_routes(best, scenario.n_deliveries, scenario.n_vehicles)


def _fits(scenario: Scenario, vehicle_idx: int, seq: list[int]) -> bool:
    """Rota respeita TODAS as restrições rígidas do veículo?"""
    r = simulate_route(scenario, vehicle_idx, seq)
    return (
        r.overload_kg == 0
        and r.over_range_km == 0
        and r.refrigeration_violations == 0
        and r.late_min == 0
        and r.overtime_min == 0
    )


def nearest_neighbor(scenario: Scenario, weights: CostWeights | None = None) -> Routes:
    """Vizinho mais próximo sequencial: cada veículo sai do depósito e vai sempre para a
    entrega viável mais próxima; quando nada mais cabe, o próximo veículo assume.
    Sobras (se houver) entram por inserção mais barata."""
    w = weights or CostWeights()
    dist = scenario.dist
    unvisited = set(range(1, scenario.n_deliveries + 1))
    routes: Routes = []
    for j in range(scenario.n_vehicles):
        seq: list[int] = []
        current = 0
        while unvisited:
            for cand in sorted(unvisited, key=lambda d: dist[current, d]):
                if _fits(scenario, j, seq + [cand]):
                    seq.append(cand)
                    unvisited.discard(cand)
                    current = cand
                    break
            else:
                break
        routes.append(seq)
    for d in sorted(unvisited):
        cheapest_insertion(scenario, routes, d, w)
    return routes


def clarke_wright(scenario: Scenario, weights: CostWeights | None = None) -> Routes:
    """Savings de Clarke & Wright (1964), versão paralela.

    Começa com uma rota por entrega e funde pares de rotas pela maior economia
    s(i,j) = d(0,i) + d(0,j) − d(i,j), desde que a rota fundida caiba em ALGUM veículo
    compatível. Depois, as rotas são atribuídas aos veículos (maiores cargas primeiro);
    rotas excedentes são desfeitas e suas entregas inseridas pelo menor custo.
    """
    w = weights or CostWeights()
    n, dist = scenario.n_deliveries, scenario.dist
    route_of = {i: i for i in range(1, n + 1)}
    routes: dict[int, list[int]] = {i: [i] for i in range(1, n + 1)}

    def fits_any(seq: list[int]) -> bool:
        return any(_fits(scenario, j, seq) for j in range(scenario.n_vehicles))

    savings = sorted(
        ((dist[0, i] + dist[0, j] - dist[i, j], i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)),
        reverse=True,
    )
    for _, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        a, b = routes[ri], routes[rj]
        # só funde se i e j estão nas pontas das respectivas rotas
        merged = None
        for x in (a, a[::-1]):
            for y in (b, b[::-1]):
                if x[-1] == i and y[0] == j:
                    merged = x + y
        if merged is None:
            for x in (b, b[::-1]):
                for y in (a, a[::-1]):
                    if x[-1] == j and y[0] == i:
                        merged = x + y
        if merged is None or not fits_any(merged):
            continue
        routes[ri] = merged
        del routes[rj]
        for d in merged:
            route_of[d] = ri

    # atribuição rota → veículo
    final: Routes = [[] for _ in scenario.vehicles]
    leftovers: list[int] = []
    pending = sorted(routes.values(), key=lambda r: -sum(scenario.delivery(d).demand_kg for d in r))
    for r in pending:
        options = [j for j in range(scenario.n_vehicles) if not final[j] and _fits(scenario, j, r)]
        if options:
            j = min(options, key=lambda j: scenario.vehicles[j].capacity_kg)
            final[j] = list(r)
        else:
            leftovers.extend(r)
    for d in leftovers:
        cheapest_insertion(scenario, final, d, w)
    return final


def nn_chromosome(scenario: Scenario, weights: CostWeights | None = None) -> list[int]:
    return join_routes(nearest_neighbor(scenario, weights), scenario.n_deliveries)
