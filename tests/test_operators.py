import numpy as np
import pytest

from rotas.ga.operators import CROSSOVERS, MUTATIONS, roulette_selection, tournament_selection


@pytest.mark.parametrize("name", list(CROSSOVERS))
def test_crossover_preserves_permutation(name):
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 30))
        p1, p2 = list(rng.permutation(n) + 1), list(rng.permutation(n) + 1)
        child = CROSSOVERS[name]([int(x) for x in p1], [int(x) for x in p2], rng)
        assert sorted(child) == list(range(1, n + 1))


def test_ox_keeps_segment_from_first_parent():
    rng = np.random.default_rng(3)
    p1, p2 = list(range(1, 11)), list(range(10, 0, -1))
    child = CROSSOVERS["ox"](p1, p2, rng)
    # o filho contém um trecho contíguo de p1 na mesma posição
    assert any(child[i : i + 3] == p1[i : i + 3] for i in range(8))


def test_crossover_identical_parents_returns_parent():
    rng = np.random.default_rng(1)
    p = [3, 1, 4, 2, 5]
    for name in CROSSOVERS:
        assert CROSSOVERS[name](p, p, rng) == p


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutation_preserves_permutation(name):
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(2, 30))
        p = [int(x) for x in rng.permutation(n) + 1]
        assert sorted(MUTATIONS[name](p, rng)) == sorted(p)


def test_inversion_reverses_a_segment():
    rng = np.random.default_rng(5)
    p = list(range(1, 9))
    q = MUTATIONS["inversion"](p, rng)
    diff = [i for i in range(len(p)) if p[i] != q[i]]
    if diff:
        i, j = diff[0], diff[-1] + 1
        assert q[i:j] == p[i:j][::-1]


def test_tournament_with_full_size_picks_best():
    rng = np.random.default_rng(0)
    pop = [[1, 2], [2, 1], [1, 2]]
    assert tournament_selection(pop, [5.0, 1.0, 3.0], rng, k=3) == [2, 1]


def test_roulette_prefers_low_cost():
    rng = np.random.default_rng(0)
    pop = [[1], [2]]
    picks = [roulette_selection(pop, [1.0, 100.0], rng)[0] for _ in range(500)]
    assert picks.count(1) > 450
