import json

import httpx
import pytest

from rotas.baselines.heuristics import nearest_neighbor
from rotas.solution import Solution
from rotas.viz.maps import route_map
from rotas.viz.roads import RoadRouter, default_router

PONTOS = [(-23.557, -46.670), (-23.540, -46.576), (-23.557, -46.670)]
CURVA = [-46.6, -23.9]  # ponto [lon, lat] inventado, fora da reta: prova que o desenho segue o OSRM


def _router(tmp_path, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RoadRouter(base_url="http://osrm.test", cache_path=tmp_path / "cache.json", pause_s=0, client=client)


def _ok(request: httpx.Request) -> httpx.Response:
    coords = [[float(v) for v in c.split(",")] for c in request.url.path.split("/")[-1].split(";")]
    geometry = [coords[0], CURVA, *coords[1:]]
    return httpx.Response(200, json={"code": "Ok", "routes": [{"distance": 12345.6, "geometry": {"coordinates": geometry}}]})


def test_route_converts_geojson_to_lat_lon(tmp_path):
    road = _router(tmp_path, _ok).route(PONTOS)
    assert road.source == "osrm" and road.km == 12.35
    assert road.points[0] == PONTOS[0] and (-23.9, -46.6) in road.points


def test_second_call_comes_from_cache_file(tmp_path):
    calls = []
    router = _router(tmp_path, lambda r: (calls.append(1), _ok(r))[1])
    router.route(PONTOS)
    again = _router(tmp_path, lambda r: pytest.fail("não devia consultar a rede")).route(PONTOS)  # novo objeto, mesmo arquivo
    assert len(calls) == 1 and again.source == "cache" and again.km == 12.35
    assert json.loads((tmp_path / "cache.json").read_text())


def test_server_error_falls_back_to_straight_line_and_stops_trying(tmp_path):
    calls = []
    router = _router(tmp_path, lambda r: (calls.append(1), httpx.Response(503))[1])
    first = router.route(PONTOS)
    second = router.route([(-23.5, -46.6), (-23.6, -46.7)])
    assert first.source == second.source == "reta" and first.points == PONTOS and first.km is None
    assert len(calls) == 1  # depois da 1ª falha não insiste


def test_network_error_falls_back(tmp_path):
    def boom(request):
        raise httpx.ConnectError("sem internet")

    assert _router(tmp_path, boom).route(PONTOS).source == "reta"


def test_no_route_only_affects_that_route(tmp_path):
    answers = iter([httpx.Response(400, json={"code": "NoRoute"}), _ok])
    router = _router(tmp_path, lambda r: (a if isinstance(a := next(answers), httpx.Response) else a(r)))
    assert router.route(PONTOS).source == "reta"
    assert router.route([(-23.5, -46.6), (-23.6, -46.7)]).source == "osrm"


def test_map_draws_the_osrm_geometry(scenario, tmp_path):
    sol = Solution(scenario, nearest_neighbor(scenario), "Vizinho mais próximo")
    html = route_map(sol, router=_router(tmp_path, _ok)).get_root().render()
    assert "-23.9" in html and "pelas ruas" in html and "8 8" not in html


def test_map_without_router_is_dashed_straight_line(scenario):
    sol = Solution(scenario, nearest_neighbor(scenario), "Vizinho mais próximo")
    html = route_map(sol, router=None).get_root().render()
    assert "8 8" in html and "linha reta" in html


def test_env_switch(monkeypatch):
    monkeypatch.setenv("ROTAS_ROADS", "off")
    assert default_router() is None
    monkeypatch.setenv("ROTAS_ROADS", "osrm")
    assert isinstance(default_router(), RoadRouter)
