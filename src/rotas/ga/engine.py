"""Algoritmo Genético para o VRP hospitalar.

Laço (a partir do código base, generalizado):

    população inicial: aleatória + fração semeada com vizinho mais próximo (+ mutações)
    para cada geração:
        avalia custo (cache por cromossomo)
        [opcional] busca local (relocate + 2-opt) nos melhores (AG memético)
        elitismo: copia os E melhores
        enquanto a nova população não estiver cheia:
            seleciona 2 pais (torneio ou roleta)
            cruza com probabilidade p_c (OX | PMX | ERX)
            muta com probabilidade p_m (troca | inversão | inserção)

Correção em relação ao código base: em ``legacy/tsp.py`` o cruzamento era chamado como
``order_crossover(parent1, parent1)`` — o filho era sempre cópia do pai e a evolução dependia
só da mutação. Aqui os dois pais são distintos.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from rotas.baselines.heuristics import nn_chromosome, random_solution
from rotas.domain import Scenario
from rotas.evaluation import CostWeights, Evaluation, evaluate_chromosome
from rotas.ga.local_search import local_search
from rotas.ga.operators import (
    CROSSOVERS,
    MUTATIONS,
    roulette_selection,
    tournament_selection,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 150
    generations: int = 1500
    crossover: str = "ox"  # ox | pmx | erx
    crossover_rate: float = 0.9
    mutation_rate: float = 0.6
    mutation_ops: tuple[str, ...] = ("swap", "inversion", "insertion")
    selection: str = "tournament"  # tournament | roulette
    tournament_size: int = 3
    elitism: int = 2
    nn_seed_fraction: float = 0.0  # fração da população inicial derivada do vizinho mais próximo
    local_search: bool = False  # AG memético: relocate + 2-opt nos melhores a cada `ls_every` gerações
    ls_top: int = 2
    ls_every: int = 10
    early_stopping: int | None = None  # gerações sem melhora antes de parar
    seed: int = 42

    def __post_init__(self) -> None:
        if self.crossover not in CROSSOVERS:
            raise ValueError(f"crossover deve ser um de {list(CROSSOVERS)}")
        if not set(self.mutation_ops) <= set(MUTATIONS):
            raise ValueError(f"mutation_ops deve estar contido em {list(MUTATIONS)}")
        if self.selection not in ("tournament", "roulette"):
            raise ValueError("selection deve ser 'tournament' ou 'roulette'")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("elitism deve estar em [0, population_size)")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationStats:
    generation: int
    best_cost: float
    mean_cost: float
    best_distance_km: float
    best_feasible: bool
    diversity: float
    elapsed_s: float


@dataclass
class GAResult:
    best_chromosome: list[int]
    best: Evaluation
    history: list[GenerationStats] = field(default_factory=list)
    n_evaluations: int = 0
    elapsed_s: float = 0.0
    config: dict = field(default_factory=dict)

    @property
    def routes(self) -> list[list[int]]:
        return [r.sequence for r in self.best.routes]


GenerationCallback = Callable[[GenerationStats, list[int]], None]


class GeneticVRP:
    def __init__(
        self,
        scenario: Scenario,
        config: GAConfig | None = None,
        weights: CostWeights | None = None,
        callbacks: Sequence[GenerationCallback] = (),
    ) -> None:
        self.scenario = scenario
        self.config = config or GAConfig()
        self.weights = weights or CostWeights()
        self.callbacks = list(callbacks)
        self.rng = np.random.default_rng(self.config.seed)
        self._cache: dict[tuple[int, ...], float] = {}
        self.n_evaluations = 0

    def cost(self, chrom: Sequence[int]) -> float:
        key = tuple(chrom)
        c = self._cache.get(key)
        if c is None:
            c = evaluate_chromosome(self.scenario, chrom, self.weights).cost
            self._cache[key] = c
            self.n_evaluations += 1
        return c

    def _mutate(self, chrom: list[int]) -> list[int]:
        if self.rng.random() < self.config.mutation_rate:
            op = self.config.mutation_ops[self.rng.integers(len(self.config.mutation_ops))]
            return MUTATIONS[op](chrom, self.rng)
        return chrom

    def initial_population(self) -> list[list[int]]:
        cfg = self.config
        pop: list[list[int]] = []
        n_seed = int(round(cfg.nn_seed_fraction * cfg.population_size))
        if n_seed:
            nn = nn_chromosome(self.scenario, self.weights)
            pop.append(nn)
            while len(pop) < n_seed:  # variações do NN para não colapsar a diversidade
                pop.append(MUTATIONS["insertion"](MUTATIONS["inversion"](nn, self.rng), self.rng))
        while len(pop) < cfg.population_size:
            pop.append(random_solution(self.scenario, self.rng))
        return pop

    def _select(self, pop, costs) -> list[int]:
        if self.config.selection == "roulette":
            return roulette_selection(pop, costs, self.rng)
        return tournament_selection(pop, costs, self.rng, self.config.tournament_size)

    def run(self) -> GAResult:
        cfg = self.config
        cross = CROSSOVERS[cfg.crossover]
        t0 = time.perf_counter()
        pop = self.initial_population()
        best_chrom: list[int] = []
        best_cost = np.inf
        stale = 0
        history: list[GenerationStats] = []

        for gen in range(cfg.generations + 1):
            costs = [self.cost(c) for c in pop]
            order = np.argsort(costs)

            if cfg.local_search and gen % cfg.ls_every == 0:
                for i in order[: cfg.ls_top]:
                    pop[i] = local_search(self.scenario, pop[i], self.weights)
                    costs[i] = self.cost(pop[i])
                order = np.argsort(costs)

            i0 = int(order[0])
            if costs[i0] < best_cost - 1e-9:
                best_cost, best_chrom, stale = costs[i0], list(pop[i0]), 0
            else:
                stale += 1

            best_ev = evaluate_chromosome(self.scenario, best_chrom, self.weights)
            stats = GenerationStats(
                generation=gen,
                best_cost=float(best_cost),
                mean_cost=float(np.mean(costs)),
                best_distance_km=float(best_ev.distance_km),
                best_feasible=best_ev.feasible,
                diversity=len({tuple(c) for c in pop}) / len(pop),
                elapsed_s=time.perf_counter() - t0,
            )
            history.append(stats)
            if gen % 25 == 0 or gen == cfg.generations:
                logger.info(
                    "geração %d: custo=%.2f dist=%.1fkm viável=%s diversidade=%.2f",
                    gen, stats.best_cost, stats.best_distance_km, stats.best_feasible, stats.diversity,
                )
            for cb in self.callbacks:
                cb(stats, best_chrom)

            if gen == cfg.generations or (cfg.early_stopping and stale >= cfg.early_stopping):
                break

            new_pop = [list(pop[i]) for i in order[: cfg.elitism]]
            while len(new_pop) < cfg.population_size:
                p1, p2 = self._select(pop, costs), self._select(pop, costs)
                child = cross(p1, p2, self.rng) if self.rng.random() < cfg.crossover_rate else list(p1)
                new_pop.append(self._mutate(child))
            pop = new_pop

        return GAResult(
            best_chromosome=best_chrom,
            best=evaluate_chromosome(self.scenario, best_chrom, self.weights),
            history=history,
            n_evaluations=self.n_evaluations,
            elapsed_s=time.perf_counter() - t0,
            config=cfg.to_dict(),
        )
