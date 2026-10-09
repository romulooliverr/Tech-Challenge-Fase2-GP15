"""Busca local usada na variante *memética* do AG.

* 2-opt intra-rota: inverte trechos [i, j] de uma rota enquanto houver melhora;
* relocate inter-rotas: retira uma entrega da sua rota e a reinsere na melhor posição de
  qualquer veículo (corrige atribuições ruins de entrega → veículo).

Como o custo é separável por rota, só as rotas alteradas são reavaliadas.
"""

from __future__ import annotations

from rotas.domain import Scenario
from rotas.evaluation import CostWeights, join_routes, route_cost, split_routes


def two_opt_route(
    scenario: Scenario, vehicle_idx: int, seq: list[int], weights: CostWeights, max_passes: int = 5
) -> list[int]:
    best, best_cost = list(seq), route_cost(scenario, vehicle_idx, seq, weights)
    for _ in range(max_passes):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best) + 1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                c = route_cost(scenario, vehicle_idx, cand, weights)
                if c < best_cost - 1e-9:
                    best, best_cost, improved = cand, c, True
        if not improved:
            break
    return best


def relocate(scenario: Scenario, routes: list[list[int]], weights: CostWeights) -> list[list[int]]:
    routes = [list(r) for r in routes]
    costs = [route_cost(scenario, j, r, weights) for j, r in enumerate(routes)]
    improved = True
    while improved:
        improved = False
        for a in range(len(routes)):
            for i in range(len(routes[a])):
                d = routes[a][i]
                removed = routes[a][:i] + routes[a][i + 1 :]
                gain_out = costs[a] - route_cost(scenario, a, removed, weights)
                best = (1e-9, None, None, None)
                for b in range(len(routes)):
                    base_b = removed if b == a else routes[b]
                    cost_b = route_cost(scenario, b, base_b, weights) if b == a else costs[b]
                    for pos in range(len(base_b) + 1):
                        cand = base_b[:pos] + [d] + base_b[pos:]
                        delta = route_cost(scenario, b, cand, weights) - cost_b
                        if gain_out - delta > best[0]:
                            best = (gain_out - delta, b, cand, removed)
                if best[1] is not None:
                    _, b, cand, removed = best
                    routes[a] = removed
                    routes[b] = cand
                    costs[a] = route_cost(scenario, a, routes[a], weights)
                    costs[b] = route_cost(scenario, b, routes[b], weights)
                    improved = True
                    break
            if improved:
                break
    return routes


def local_search(scenario: Scenario, chromosome: list[int], weights: CostWeights) -> list[int]:
    routes = split_routes(chromosome, scenario.n_deliveries, scenario.n_vehicles)
    routes = relocate(scenario, routes, weights)
    routes = [two_opt_route(scenario, j, r, weights) for j, r in enumerate(routes)]
    return join_routes(routes, scenario.n_deliveries)
