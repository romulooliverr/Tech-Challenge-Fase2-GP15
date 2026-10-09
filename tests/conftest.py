import os

import pytest

os.environ["LLM_PROVIDER"] = "mock"  # testes nunca chamam a API real

from rotas.scenario import generate_scenario  # noqa: E402


@pytest.fixture(scope="session")
def scenario():
    return generate_scenario(n_deliveries=15, seed=7)


@pytest.fixture(scope="session")
def big_scenario():
    return generate_scenario(n_deliveries=40, seed=42)
