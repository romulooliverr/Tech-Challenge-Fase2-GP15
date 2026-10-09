"""Baseline "estado da arte": Google OR-Tools Routing (busca local guiada).

Modela as mesmas restrições do AG:

* capacidade por veículo (dimensão de carga);
* autonomia por veículo (dimensão de distância com limite por veículo);
* tempo com velocidade diferente por veículo, tempo de serviço, janelas dos pacientes
  domiciliares e fim de turno (restrições suaves, como no AG);
* cadeia fria: entregas refrigeradas só podem ir em veículos refrigerados;
* prioridade: limite superior suave de horário para entregas críticas/urgentes.

A solução é reavaliada pela MESMA função de custo do AG, para comparação justa.
"""

from __future__ import annotations

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from rotas.domain import Priority, Scenario

SCALE = 100  # OR-Tools trabalha com inteiros: km*100, kg*10, minutos inteiros


def solve_ortools(scenario: Scenario, time_limit_s: int = 10) -> list[list[int]]:
    n, k = scenario.n_deliveries, scenario.n_vehicles
    manager = pywrapcp.RoutingIndexManager(n + 1, k, 0)
    routing = pywrapcp.RoutingModel(manager)
    dist = scenario.dist

    def dist_cb(i, j):
        return int(dist[manager.IndexToNode(i), manager.IndexToNode(j)] * SCALE)

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)
    routing.SetFixedCostOfAllVehicles(int(5 * SCALE))

    # autonomia
    routing.AddDimensionWithVehicleCapacity(
        dist_idx, 0, [int(v.max_range_km * SCALE) for v in scenario.vehicles], True, "Distance"
    )

    # capacidade
    def demand_cb(i):
        node = manager.IndexToNode(i)
        return 0 if node == 0 else int(scenario.delivery(node).demand_kg * 10)

    dem_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_idx, 0, [int(v.capacity_kg * 10) for v in scenario.vehicles], True, "Load"
    )

    # tempo (trânsito depende da velocidade de cada veículo)
    def make_time_cb(speed_kmh: float):
        def cb(i, j):
            a, b = manager.IndexToNode(i), manager.IndexToNode(j)
            service = 0 if a == 0 else scenario.delivery(a).service_min
            return int(round(service + dist[a, b] / speed_kmh * 60))

        return cb

    time_idxs = [routing.RegisterTransitCallback(make_time_cb(v.speed_kmh)) for v in scenario.vehicles]
    horizon = 24 * 60
    routing.AddDimensionWithVehicleTransits(time_idxs, 600, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for v in range(k):
        time_dim.CumulVar(routing.Start(v)).SetValue(scenario.shift_start_min)
        time_dim.SetCumulVarSoftUpperBound(routing.End(v), scenario.shift_end_min, 10 * SCALE)

    not_refrigerated = [v for v in range(k) if not scenario.vehicles[v].refrigerated]
    for d in scenario.deliveries:
        idx = manager.NodeToIndex(d.id)
        if d.window:
            time_dim.CumulVar(idx).SetMin(d.window[0])
            time_dim.SetCumulVarSoftUpperBound(idx, d.window[1], 10 * SCALE)
        if d.priority == Priority.CRITICA:
            time_dim.SetCumulVarSoftUpperBound(idx, scenario.shift_start_min + 60, int(0.1 * SCALE))
        elif d.priority == Priority.URGENTE:
            time_dim.SetCumulVarSoftUpperBound(idx, scenario.shift_start_min + 60, int(0.033 * SCALE))
        if d.refrigerated:
            routing.VehicleVar(idx).RemoveValues(not_refrigerated)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools não encontrou solução viável para o cenário.")

    routes: list[list[int]] = []
    for v in range(k):
        seq, idx = [], routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node:
                seq.append(node)
            idx = solution.Value(routing.NextVar(idx))
        routes.append(seq)
    return routes
