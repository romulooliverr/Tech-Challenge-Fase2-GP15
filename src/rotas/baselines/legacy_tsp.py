"""Roda o AG do código base (``legacy/genetic_algorithm.py``) sem modificações algorítmicas.

O código base resolve um TSP puro: um único "caixeiro", sem capacidade, autonomia,
prioridade, cadeia fria ou janelas. Para comparar no cenário hospitalar, a volta inteira é
atribuída à maior van — e avaliada pela mesma função de custo, o que evidencia as violações.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

from rotas.domain import Scenario

LEGACY_DIR = Path(__file__).resolve().parents[3] / "legacy"


def _load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_ga", LEGACY_DIR / "genetic_algorithm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def solve_legacy_tsp(
    scenario: Scenario,
    population_size: int = 100,
    generations: int = 1500,
    mutation_probability: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    ga = _load_legacy()
    random.seed(seed)
    # o código base trabalha com tuplas de coordenadas; o depósito entra como mais uma "cidade"
    coords = {(scenario.depot.lat, scenario.depot.lon): 0}
    coords.update({(d.location.lat, d.location.lon): d.id for d in scenario.deliveries})
    cities = list(coords)

    population = ga.generate_random_population(cities, population_size)
    for _ in range(generations):  # mesmo laço do __main__ de legacy/genetic_algorithm.py
        fitness = [ga.calculate_fitness(ind) for ind in population]
        population, fitness = ga.sort_population(population, fitness)
        new_population = [population[0]]
        while len(new_population) < population_size:
            p1, p2 = random.choices(population[:10], k=2)
            child = ga.mutate(ga.order_crossover(p1, p2), mutation_probability)
            new_population.append(child)
        population = new_population
    best = min(population, key=ga.calculate_fitness)

    tour = [coords[c] for c in best]
    start = tour.index(0)
    tour = tour[start + 1 :] + tour[:start]  # gira a volta para começar/terminar no depósito
    biggest = max(range(scenario.n_vehicles), key=lambda j: scenario.vehicles[j].capacity_kg)
    routes: list[list[int]] = [[] for _ in scenario.vehicles]
    routes[biggest] = tour
    return routes
