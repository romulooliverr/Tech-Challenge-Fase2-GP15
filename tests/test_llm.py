import pytest

from rotas.baselines.heuristics import nearest_neighbor
from rotas.llm import prompts
from rotas.llm.assistant import RouteAssistant, context_as_tables
from rotas.llm.provider import GeminiProvider, MockProvider, QuotaExhausted, get_provider
from rotas.llm.quality import evaluate_driver_text, numeric_faithfulness
from rotas.solution import Solution, route_details


@pytest.fixture()
def assistant(scenario):
    sol = Solution(scenario, nearest_neighbor(scenario), "Vizinho mais próximo")
    return RouteAssistant(sol, provider=MockProvider())


def test_get_provider_mock():
    assert get_provider("mock").name == "mock"
    with pytest.raises(ValueError):
        get_provider("inexistente")


def test_prompt_contains_data_block(assistant):
    data = assistant.route_data(1)
    text = prompts.render(prompts.DRIVER_V1, data)
    assert "<dados>" in text and data["veiculo"] in text


def test_mock_instructions_list_stops_in_order(assistant):
    vid = next(r["veiculo_id"] for r in assistant.context["rotas"] if r["paradas"])
    route = assistant.route_data(vid)
    text = assistant.driver_instructions(vid)
    q = evaluate_driver_text(text, route)
    assert q["cobertura_ordem"]["cobertura"] == 1.0
    assert q["cobertura_ordem"]["ordem_correta"]


def test_numeric_faithfulness_flags_hallucinated_numbers():
    data = {"distancia_km": 42.5, "chegada": "09:15"}
    assert numeric_faithfulness("Distância 42.5 km, chegada 09:15", data)["score"] == 1.0
    bad = numeric_faithfulness("Distância 57.3 km, chegada 10:40", data)
    assert bad["score"] == 0.0 and "57.3" in bad["nao_suportados"]


def test_quality_detects_missing_stop_and_highlight(assistant):
    vid = next(r["veiculo_id"] for r in assistant.context["rotas"] if len(r["paradas"]) >= 2)
    route = assistant.route_data(vid)
    text = "## Paradas\n" + f"**1. {route['paradas'][0]['destino']}**"
    q = evaluate_driver_text(text, route)
    assert q["cobertura_ordem"]["cobertura"] < 1.0
    assert q["estrutura"]["score"] < 1.0


def test_tools_search_is_accent_insensitive(assistant):
    tools = {t.__name__: t for t in assistant.tools()}
    any_stop = next(p for r in assistant.context["rotas"] for p in r["paradas"])
    district = any_stop["bairro"]
    plain = district.replace("ã", "a").replace("é", "e").replace("ó", "o").replace("â", "a").upper()
    assert tools["buscar_entrega"](plain)["resultados"]
    crit = tools["entregas_por_prioridade"]("critica")
    assert crit["total"] == sum(p["prioridade"] == "crítica" for r in assistant.context["rotas"] for p in r["paradas"])


def test_context_tables_have_index_and_all_stops(assistant):
    text = context_as_tables(assistant.context)
    assert "ÍNDICE" in text
    for r in assistant.context["rotas"]:
        for p in r["paradas"]:
            assert p["destino"] in text


def test_chat_modes_with_mock(assistant):
    assert assistant.start_chat("tools").send("qual a distância total?").startswith("[MOCK]")
    assert assistant.start_chat("context").send("oi")


def test_route_details_consistent_with_evaluation(scenario):
    sol = Solution(scenario, nearest_neighbor(scenario), "nn")
    details = route_details(scenario, sol.evaluate())
    assert sum(len(r["paradas"]) for r in details) == scenario.n_deliveries


def test_gemini_falls_back_when_daily_quota_exhausted(monkeypatch):
    from google.genai import errors

    calls = []

    def fake(model):
        calls.append(model)
        if model == "modelo-a":
            raise errors.ClientError(429, {"error": {"code": 429, "message": "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier", "status": "RESOURCE_EXHAUSTED"}})
        return "ok"

    GeminiProvider._exhausted.clear()
    p = GeminiProvider(model="modelo-a", api_key="x", fallbacks=["modelo-b"])
    assert p._call(fake) == "ok" and calls == ["modelo-a", "modelo-b"] and p.last_model == "modelo-b"
    GeminiProvider._exhausted.add("modelo-b")
    with pytest.raises(QuotaExhausted):
        p._call(fake)
    GeminiProvider._exhausted.clear()
