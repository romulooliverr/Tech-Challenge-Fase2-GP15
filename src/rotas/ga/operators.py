"""Operadores genéticos para permutações (rotas).

Todos os operadores preservam a propriedade de permutação: cada entrega e cada separador
aparece exatamente uma vez no filho — requisito para que a rota decodificada seja válida.

Evolução em relação ao código base (``legacy/genetic_algorithm.py``):

* OX reescrito com índices inteiros e conjunto (O(n)), em vez de ``gene not in child`` sobre
  tuplas de coordenadas (O(n²) e quebrava com dois pontos de mesma coordenada);
* novos cruzamentos: PMX e ERX (Edge Recombination, que preserva adjacências — ótimo p/ TSP);
* mutações além da troca de vizinhos: troca arbitrária, inversão de segmento (equivalente a um
  movimento 2-opt — era o TODO do código base) e inserção (realoca uma entrega/separador);
* seleção por torneio e por roleta (a roleta 1/custo é a do código base).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Perm = list[int]

# ---------------------------------------------------------------------------- seleção


def tournament_selection(
    population: Sequence[Perm], costs: Sequence[float], rng: np.random.Generator, k: int = 3
) -> Perm:
    idx = rng.choice(len(population), size=min(k, len(population)), replace=False)
    return list(population[min(idx, key=lambda i: costs[i])])


def roulette_selection(
    population: Sequence[Perm], costs: Sequence[float], rng: np.random.Generator
) -> Perm:
    """Probabilidade proporcional a 1/custo (estratégia do código base)."""
    w = 1.0 / np.asarray(costs, dtype=float)
    return list(population[rng.choice(len(population), p=w / w.sum())])


# ---------------------------------------------------------------------------- cruzamento


def order_crossover(p1: Perm, p2: Perm, rng: np.random.Generator) -> Perm:
    """OX: copia um trecho de p1 e completa com os genes restantes na ordem em que aparecem em p2."""
    n = len(p1)
    a, b = sorted(rng.choice(n + 1, size=2, replace=False))
    middle = p1[a:b]
    taken = set(middle)
    rest = [g for g in p2 if g not in taken]
    return rest[:a] + middle + rest[a:]


def pmx_crossover(p1: Perm, p2: Perm, rng: np.random.Generator) -> Perm:
    """PMX: copia um trecho de p1; os conflitos são resolvidos pelo mapeamento trecho p1 ↔ p2."""
    n = len(p1)
    a, b = sorted(rng.choice(n + 1, size=2, replace=False))
    child: list[int | None] = [None] * n
    child[a:b] = p1[a:b]
    pos_in_p2 = {g: i for i, g in enumerate(p2)}
    in_middle = set(p1[a:b])
    for i in range(a, b):
        g = p2[i]
        if g in in_middle:
            continue
        j = i
        while a <= j < b:  # segue o mapeamento p1[j] -> posição em p2 até sair do trecho
            j = pos_in_p2[p1[j]]
        child[j] = g
    for i in range(n):
        if child[i] is None:
            child[i] = p2[i]
    return child  # type: ignore[return-value]


def edge_recombination(p1: Perm, p2: Perm, rng: np.random.Generator) -> Perm:
    """ERX: constrói o filho preferindo vizinhos (arestas) presentes em algum dos pais."""
    n = len(p1)
    adj: dict[int, set[int]] = {g: set() for g in p1}
    for p in (p1, p2):
        for i, g in enumerate(p):
            adj[g].update((p[i - 1], p[(i + 1) % n]))
    current = p1[0] if rng.random() < 0.5 else p2[0]
    child = [current]
    remaining = set(p1) - {current}
    while remaining:
        for s in adj.values():
            s.discard(current)
        candidates = adj.pop(current)
        if candidates:
            fewest = min(len(adj[c]) for c in candidates)
            options = sorted(c for c in candidates if len(adj[c]) == fewest)
        else:
            options = sorted(remaining)
        current = options[rng.integers(len(options))]
        child.append(current)
        remaining.discard(current)
    return child


CROSSOVERS: dict[str, Callable[[Perm, Perm, np.random.Generator], Perm]] = {
    "ox": order_crossover,
    "pmx": pmx_crossover,
    "erx": edge_recombination,
}

# ---------------------------------------------------------------------------- mutação


def swap_mutation(p: Perm, rng: np.random.Generator) -> Perm:
    i, j = rng.choice(len(p), size=2, replace=False)
    q = list(p)
    q[i], q[j] = q[j], q[i]
    return q


def inversion_mutation(p: Perm, rng: np.random.Generator) -> Perm:
    """Inverte um trecho (movimento 2-opt): desfaz cruzamentos de arestas na rota."""
    i, j = sorted(rng.choice(len(p) + 1, size=2, replace=False))
    return p[:i] + p[i:j][::-1] + p[j:]


def insertion_mutation(p: Perm, rng: np.random.Generator) -> Perm:
    """Remove um gene e o reinsere em outra posição (move entrega de rota/veículo)."""
    q = list(p)
    g = q.pop(int(rng.integers(len(q))))
    q.insert(int(rng.integers(len(q) + 1)), g)
    return q


MUTATIONS: dict[str, Callable[[Perm, np.random.Generator], Perm]] = {
    "swap": swap_mutation,
    "inversion": inversion_mutation,
    "insertion": insertion_mutation,
}
