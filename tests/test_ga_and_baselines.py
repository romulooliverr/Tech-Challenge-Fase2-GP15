import numpy as np
import pytest

from rotas.baselines.heuristics import clarke_wright, nearest_neighbor, random_solution
from rotas.baselines.legacy_tsp import solve_legacy_tsp
from rotas.baselines.ortools_solver import solve_ortools
from rotas.domain import Scenario
from rotas.evaluation import evaluate_chromosome, evaluate_routes, join_routes
from rotas.ga.engine import GAConfig, GeneticVRP
from rotas.ga.local_search import local_search
from rotas.scenario import generate_scenario


def _covers_all(routes, n):
    flat = [d for r in routes for d in r]
    return sorted(flat) == list(range(1, n + 1))


def test_scenario_is_reproducible_and_serializable(tmp_path):
    a, b = generate_scenario(20, seed=3), generate_scenario(20, seed=3)
    assert a.to_dict() == b.to_dict()
    a.save(tmp_path / "s.json")
    c = Scenario.load(tmp_path / "s.json")
    assert c.to_dict() == a.to_dict()
    assert np.allclose(c.dist, a.dist)


@pytest.mark.parametrize("solver", [nearest_neighbor, clarke_wright])
def test_heuristics_serve_every_delivery_once(scenario, solver):
    routes = solver(scenario)
    assert len(routes) == scenario.n_vehicles
    assert _covers_all(routes, scenario.n_deliveries)


def test_ortools_serves_every_delivery(scenario):
    routes = solve_ortools(scenario, time_limit_s=2)
    assert _covers_all(routes, scenario.n_deliveries)
    assert evaluate_routes(scenario, routes).routes  # avaliável pela mesma função


def test_legacy_tsp_runs_and_uses_single_vehicle(scenario):
    routes = solve_legacy_tsp(scenario, population_size=20, generations=20)
    assert _covers_all(routes, scenario.n_deliveries)
    assert sum(bool(r) for r in routes) == 1


def test_ga_is_reproducible_and_beats_random(scenario):
    cfg = GAConfig(population_size=40, generations=80, seed=1)
    r1, r2 = GeneticVRP(scenario, cfg).run(), GeneticVRP(scenario, cfg).run()
    assert r1.best_chromosome == r2.best_chromosome
    assert sorted(r1.best_chromosome) == list(range(1, scenario.n_deliveries + scenario.n_vehicles))
    rng = np.random.default_rng(0)
    random_best = min(evaluate_chromosome(scenario, random_solution(scenario, rng)).cost for _ in range(200))
    assert r1.best.cost < random_best


def test_ga_best_cost_never_increases(scenario):
    res = GeneticVRP(scenario, GAConfig(population_size=30, generations=50, seed=2)).run()
    costs = [h.best_cost for h in res.history]
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))


def test_local_search_never_worsens(scenario):
    rng = np.random.default_rng(4)
    for _ in range(5):
        chrom = random_solution(scenario, rng)
        improved = local_search(scenario, chrom, GeneticVRP(scenario).weights)
        assert sorted(improved) == sorted(chrom)
        assert evaluate_chromosome(scenario, improved).cost <= evaluate_chromosome(scenario, chrom).cost + 1e-9


def test_nn_seed_is_valid_chromosome(scenario):
    chrom = join_routes(nearest_neighbor(scenario), scenario.n_deliveries)
    assert sorted(chrom) == list(range(1, scenario.n_deliveries + scenario.n_vehicles))


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        GAConfig(crossover="xyz")
    with pytest.raises(ValueError):
        GAConfig(population_size=10, elitism=10)
