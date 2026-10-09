"""Experimentos do AG (várias configurações × várias seeds) + comparativo com outros métodos.

Gera em ``results/``:

* ``experiments_runs.csv``      — uma linha por execução (configuração, seed);
* ``experiments_summary.csv``   — média/desvio/melhor por configuração;
* ``methods_comparison.csv``    — AG vs. aleatório, vizinho mais próximo, Clarke-Wright,
                                   OR-Tools e o código base original (TSP);
* ``convergence.png``, ``boxplot_cost.png``, ``methods_comparison.png``;
* ``solutions/*.json`` e ``mapa_*.html`` — soluções e mapas.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from rotas.baselines.heuristics import clarke_wright, nearest_neighbor, random_search
from rotas.baselines.legacy_tsp import solve_legacy_tsp
from rotas.baselines.ortools_solver import solve_ortools
from rotas.domain import Scenario
from rotas.evaluation import evaluate_routes, summarize
from rotas.ga.engine import GAConfig, GeneticVRP
from rotas.scenario import generate_scenario
from rotas.solution import Solution
from rotas.viz.maps import save_route_map
from rotas.viz.plots import plot_boxplot, plot_convergence, plot_method_comparison

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Cada experimento altera UM fator em relação à configuração base (análise ceteris paribus).
EXPERIMENTS: dict[str, dict] = {
    "E1_base": {},
    "E2_pop_50": {"population_size": 50},
    "E3_pop_300": {"population_size": 300},
    "E4_mutacao_0.1": {"mutation_rate": 0.1},
    "E5_cx_pmx": {"crossover": "pmx"},
    "E6_cx_erx": {"crossover": "erx"},
    "E7_selecao_roleta": {"selection": "roulette"},
    "E8_semente_vizinho": {"nn_seed_fraction": 0.1},
    "E9_memetico": {"local_search": True},
}


def _run_ga(name: str, overrides: dict, scenario_dict: dict, seed: int, generations: int) -> dict:
    scenario = Scenario.from_dict(scenario_dict)
    cfg = GAConfig(**{**overrides, "seed": seed, "generations": generations})
    res = GeneticVRP(scenario, cfg).run()
    s = summarize(scenario, res.best)
    return {
        "row": {
            "experiment": name,
            "seed": seed,
            "cost": res.best.cost,
            "distance_km": res.best.distance_km,
            "feasible": res.best.feasible,
            "vehicles_used": res.best.vehicles_used,
            "critical_mean_h": s["tempo_medio_ate_criticas_h"],
            "evaluations": res.n_evaluations,
            "time_s": res.elapsed_s,
            **{f"cfg_{k}": v for k, v in overrides.items()},
        },
        "history": [asdict(h) for h in res.history],
        "routes": res.routes,
    }


def run_experiments(
    scenario: Scenario,
    seeds: list[int],
    generations: int,
    experiments: dict[str, dict] | None = None,
    workers: int | None = None,
) -> tuple[pd.DataFrame, dict[str, list], dict[str, tuple[float, list]]]:
    experiments = experiments or EXPERIMENTS
    sd = scenario.to_dict()
    jobs = [(n, o, sd, s, generations) for n, o in experiments.items() for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        outputs = list(ex.map(_run_ga, *zip(*jobs)))
    rows, histories, best = [], {}, {}
    for (name, *_), out in zip(jobs, outputs):
        rows.append(out["row"])
        histories.setdefault(name, []).append(out["history"])
        if name not in best or out["row"]["cost"] < best[name][0]:
            best[name] = (out["row"]["cost"], out["routes"])
    return pd.DataFrame(rows), histories, best


def run_baselines(scenario: Scenario, ortools_time_s: int, legacy_generations: int) -> dict[str, tuple[list, float]]:
    methods = {
        "Aleatório (melhor de 2000)": lambda: random_search(scenario, 2000),
        "Vizinho mais próximo": lambda: nearest_neighbor(scenario),
        "Clarke-Wright (savings)": lambda: clarke_wright(scenario),
        f"OR-Tools (GLS {ortools_time_s}s)": lambda: solve_ortools(scenario, ortools_time_s),
        "Código base (TSP, 1 veículo)": lambda: solve_legacy_tsp(scenario, generations=legacy_generations),
    }
    out = {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        routes = fn()
        out[name] = (routes, time.perf_counter() - t0)
        logger.info("baseline %s concluído em %.1fs", name, out[name][1])
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Experimentos do AG + comparativo de métodos")
    ap.add_argument("--deliveries", type=int, default=40)
    ap.add_argument("--scenario-seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=5, help="nº de seeds por configuração")
    ap.add_argument("--generations", type=int, default=1500)
    ap.add_argument("--ortools-time", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="rodada curta (smoke test)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    if args.quick:
        args.seeds, args.generations, args.ortools_time = 2, 200, 5

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = args.out
    (out / "solutions").mkdir(parents=True, exist_ok=True)
    scenario = generate_scenario(args.deliveries, args.scenario_seed)
    scenario.save(out / "scenario.json")

    t0 = time.perf_counter()
    runs, histories, best = run_experiments(scenario, list(range(1, args.seeds + 1)), args.generations)
    logger.info("experimentos do AG concluídos em %.1fs", time.perf_counter() - t0)
    runs.to_csv(out / "experiments_runs.csv", index=False)
    summary = (
        runs.groupby("experiment")
        .agg(
            cost_mean=("cost", "mean"),
            cost_std=("cost", "std"),
            cost_best=("cost", "min"),
            distance_mean=("distance_km", "mean"),
            feasible_rate=("feasible", "mean"),
            critical_mean_h=("critical_mean_h", "mean"),
            evaluations_mean=("evaluations", "mean"),
            time_s_mean=("time_s", "mean"),
        )
        .sort_values("cost_mean")
        .round(3)
    )
    summary.to_csv(out / "experiments_summary.csv")
    plot_convergence(histories, out / "convergence.png")
    plot_boxplot(runs, out / "boxplot_cost.png")

    # ---- comparativo de métodos: melhor configuração do AG vs. baselines
    best_exp = summary.index[0]
    methods: dict[str, tuple[list, float]] = {
        f"AG — {best_exp} (melhor de {args.seeds} seeds)": (
            best[best_exp][1],
            float(runs.loc[runs.experiment == best_exp, "time_s"].mean()),
        ),
        "AG — E1_base (melhor seed)": (
            best["E1_base"][1],
            float(runs.loc[runs.experiment == "E1_base", "time_s"].mean()),
        ),
    }
    methods.update(run_baselines(scenario, args.ortools_time, legacy_generations=args.generations))
    rows = []
    for name, (routes, secs) in methods.items():
        ev = evaluate_routes(scenario, routes)
        rows.append({"metodo": name, "tempo_s": round(secs, 2), **{k: v for k, v in summarize(scenario, ev).items() if k != "violacoes"},
                     **{f"viol_{k}": v for k, v in ev.violations().items()}})
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")[:40]
        sol = Solution(scenario, routes, name, {"tempo_s": secs})
        sol.save(out / "solutions" / f"{slug}.json")
    comp = pd.DataFrame(rows).sort_values("custo_objetivo")
    comp.to_csv(out / "methods_comparison.csv", index=False)
    plot_method_comparison(comp, out / "methods_comparison.png")

    best_name = comp.iloc[0]["metodo"]
    ga_name = next(n for n in methods if n.startswith("AG — " + best_exp))
    for name, fname in [(ga_name, "mapa_ag.html"), ("Vizinho mais próximo", "mapa_vizinho_mais_proximo.html"),
                        ("Código base (TSP, 1 veículo)", "mapa_codigo_base.html")]:
        save_route_map(Solution(scenario, methods[name][0], name), out / fname, title=name)
    Solution(scenario, methods[ga_name][0], ga_name).save(out / "best_solution.json")

    (out / "run_info.json").write_text(json.dumps(vars(args) | {"best_overall": best_name, "best_ga": ga_name}, default=str, indent=2, ensure_ascii=False))
    print("\n=== Resumo dos experimentos do AG ===")
    print(summary.to_string())
    print("\n=== Comparativo de métodos ===")
    print(comp[["metodo", "custo_objetivo", "distancia_total_km", "viavel", "tempo_medio_ate_criticas_h", "veiculos_usados", "tempo_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
