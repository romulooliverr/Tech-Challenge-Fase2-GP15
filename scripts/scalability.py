"""Escalabilidade: AG memético × OR-Tools × heurísticas em cenários de tamanhos diferentes.

Uso: python scripts/scalability.py  →  results/scalability.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from rotas.baselines.heuristics import clarke_wright, nearest_neighbor
from rotas.baselines.ortools_solver import solve_ortools
from rotas.evaluation import evaluate_routes
from rotas.ga.engine import GAConfig, GeneticVRP
from rotas.scenario import default_fleet, generate_scenario
from rotas.domain import Vehicle

OUT = Path(__file__).resolve().parents[1] / "results" / "scalability.csv"


def fleet_for(n: int) -> list[Vehicle]:
    """Frota proporcional à demanda: 5 veículos até 40 entregas, 8 para 80."""
    base = default_fleet()
    if n <= 40:
        return base
    extra = [
        Vehicle(6, "Van Refrigerada 6", "van", 400, 180, 22, True, 1.8),
        Vehicle(7, "Van Carga 7", "van", 600, 200, 22, False, 1.5),
        Vehicle(8, "Moto 8", "moto", 25, 120, 32, False, 0.5),
    ]
    return base + extra


def main() -> None:
    rows = []
    for n in (20, 40, 80):
        sc = generate_scenario(n, seed=7, vehicles=fleet_for(n))
        methods = {
            "Vizinho mais próximo": lambda: nearest_neighbor(sc),
            "Clarke-Wright": lambda: clarke_wright(sc),
            "OR-Tools (30s)": lambda: solve_ortools(sc, 30),
            "AG memético": lambda: GeneticVRP(sc, GAConfig(local_search=True, seed=1)).run().routes,
        }
        for name, fn in methods.items():
            t0 = time.perf_counter()
            ev = evaluate_routes(sc, fn())
            rows.append({"entregas": n, "veiculos": sc.n_vehicles, "metodo": name, "custo": round(ev.cost, 1),
                         "distancia_km": round(ev.distance_km, 1), "viavel": ev.feasible,
                         "tempo_s": round(time.perf_counter() - t0, 1)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.pivot(index="metodo", columns="entregas", values="custo"))


if __name__ == "__main__":
    main()
