import pytest

from rotas.domain import Delivery, Location, Priority, Scenario, Vehicle
from rotas.evaluation import (
    CostWeights,
    evaluate_chromosome,
    evaluate_routes,
    join_routes,
    route_cost,
    split_routes,
)
from rotas.geo import haversine_km
from rotas.scenario import DEPOT


def _mini(window=None, refrigerated=False, demand=10.0, speed=60.0, capacity=100.0):
    """Depósito + 2 entregas em linha reta a leste do depósito; 1 veículo."""
    deliveries = [
        Delivery(1, Location("A", DEPOT.lat, DEPOT.lon + 0.05, "ubs", "X"), demand, Priority.REGULAR, refrigerated, 0, (), window),
        Delivery(2, Location("B", DEPOT.lat, DEPOT.lon + 0.10, "ubs", "X"), demand, Priority.CRITICA, False, 0, ()),
    ]
    vehicles = [Vehicle(1, "V1", "van", capacity, 500, speed, False, 1.0), Vehicle(2, "V2", "moto", 5, 100, speed, False, 1.0)]
    return Scenario(DEPOT, deliveries, vehicles, road_factor=1.0)


def test_haversine_known_distance():
    # São Paulo (Sé) → Campinas ≈ 84 km em linha reta
    assert haversine_km(-23.5505, -46.6333, -22.9099, -47.0626) == pytest.approx(84, abs=3)


def test_split_join_roundtrip():
    chrom = [3, 1, 7, 5, 2, 8, 6, 4]
    routes = split_routes(chrom, n_deliveries=6, n_vehicles=3)
    assert routes == [[3, 1], [5, 2], [6, 4]]
    assert join_routes(routes, 6) == chrom


def test_split_rejects_wrong_number_of_separators():
    with pytest.raises(ValueError):
        split_routes([1, 2, 3], n_deliveries=3, n_vehicles=2)


def test_route_distance_out_and_back():
    sc = _mini()
    ev = evaluate_routes(sc, [[1, 2], []])
    leg = sc.dist[0, 2]
    assert ev.routes[0].distance_km == pytest.approx(2 * leg)
    assert ev.feasible and ev.vehicles_used == 1


def test_capacity_violation_is_penalized():
    sc = _mini(demand=60)
    ev = evaluate_routes(sc, [[1, 2], []])
    assert ev.routes[0].overload_kg == pytest.approx(20)
    assert not ev.feasible
    assert ev.penalty == pytest.approx(CostWeights().overload_per_kg * 20)


def test_refrigerated_item_on_non_refrigerated_vehicle():
    sc = _mini(refrigerated=True)
    ev = evaluate_routes(sc, [[1, 2], []])
    assert ev.routes[0].refrigeration_violations == 1


def test_time_window_wait_and_lateness():
    sc = _mini(window=(10 * 60, 11 * 60))  # chega ~08:05, espera até 10:00
    stop = evaluate_routes(sc, [[1], [2]]).routes[0].stops[0]
    assert stop.start_service_min == 600 and stop.late_min == 0
    sc_late = _mini(window=(6 * 60, 7 * 60))  # janela já fechou quando o turno começa
    assert evaluate_routes(sc_late, [[1], [2]]).routes[0].stops[0].late_min > 0


def test_priority_rewards_serving_critical_first():
    sc = _mini()
    first = evaluate_routes(sc, [[2, 1], []])
    last = evaluate_routes(sc, [[1, 2], []])
    assert first.priority_cost < last.priority_cost


def test_cost_is_sum_of_route_costs(big_scenario):
    import numpy as np

    rng = np.random.default_rng(0)
    n, k = big_scenario.n_deliveries, big_scenario.n_vehicles
    chrom = [int(x) for x in rng.permutation(np.arange(1, n + k))]
    ev = evaluate_chromosome(big_scenario, chrom)
    routes = split_routes(chrom, n, k)
    total = sum(route_cost(big_scenario, j, r) for j, r in enumerate(routes))
    assert ev.cost == pytest.approx(total)
