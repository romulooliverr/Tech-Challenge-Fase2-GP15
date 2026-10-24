"""Trajeto pelas ruas para o desenho dos mapas (OSRM).

O otimizador NÃO usa este módulo: ele continua calculando distâncias com Haversine × 1,3
(``rotas.geo``). Aqui só se busca, para cada rota já decidida, o caminho que um veículo faria
pelas ruas, para desenhar no mapa em vez de uma linha reta entre as paradas.

* Serviço: OSRM (``OSRM_URL``, padrão o servidor público de demonstração, sem chave).
* Cache: ``results/road_cache.json`` (ou ``ROTAS_ROAD_CACHE``). Cada trajeto é buscado uma vez só;
  depois o mapa abre sem internet. Vale versionar esse arquivo.
* Falha de rede, limite de uso (429) ou erro do servidor: o resto do mapa usa linha reta tracejada,
  sem derrubar nada. ``ROTAS_ROADS=off`` desliga a consulta.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_URL = "https://router.project-osrm.org"
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "results" / "road_cache.json"
USER_AGENT = "rotas-tech-challenge/0.1 (projeto academico FIAP)"

Point = tuple[float, float]  # (lat, lon)


@dataclass
class RoadPath:
    points: list[Point]  # (lat, lon), na ordem do trajeto
    km: float | None  # comprimento pelas ruas; None quando é só linha reta
    source: str  # "osrm" | "cache" | "reta"

    @property
    def is_road(self) -> bool:
        return self.source != "reta"


class RoadRouter:
    """Cliente mínimo do OSRM (``/route/v1``) com cache em arquivo."""

    def __init__(self, base_url: str | None = None, profile: str | None = None, cache_path: Path | str | None = None,
                 timeout: float = 8.0, pause_s: float = 1.0, client=None) -> None:
        import httpx

        self.base_url = (base_url or os.getenv("OSRM_URL", DEFAULT_URL)).rstrip("/")
        self.profile = profile or os.getenv("OSRM_PROFILE", "driving")
        self.cache_path = Path(cache_path or os.getenv("ROTAS_ROAD_CACHE") or DEFAULT_CACHE)
        self.pause_s = pause_s  # o servidor público pede no máximo ~1 requisição por segundo
        self._http = client or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
        self._cache: dict | None = None
        self._offline = False  # após uma falha de rede/limite, não insiste no resto da sessão
        self._last_request = 0.0

    # ------------------------------------------------------------------ cache
    def _load(self) -> dict:
        if self._cache is None:
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        return self._cache

    def _save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False))
            tmp.replace(self.cache_path)
        except OSError as e:  # cache é só otimização
            logger.warning("Não consegui gravar o cache de ruas (%s)", e)

    def _key(self, points: Sequence[Point]) -> str:
        raw = self.profile + "|" + ";".join(f"{lat:.6f},{lon:.6f}" for lat, lon in points)
        return hashlib.sha1(raw.encode()).hexdigest()

    # ------------------------------------------------------------------ consulta
    def route(self, points: Sequence[Point]) -> RoadPath:
        """Trajeto pelas ruas passando pelos pontos, em ordem. Nunca levanta exceção de rede."""
        points = list(points)
        straight = RoadPath(points, None, "reta")
        if len(points) < 2:
            return straight
        key = self._key(points)
        hit = self._load().get(key)
        if hit:
            return RoadPath([tuple(p) for p in hit["points"]], hit["km"], "cache")
        if self._offline:
            return straight

        import httpx

        coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
        wait = self.pause_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            self._last_request = time.monotonic()
            r = self._http.get(
                f"{self.base_url}/route/v1/{self.profile}/{coords}",
                params={"overview": "full", "geometries": "geojson"},
            )
        except httpx.HTTPError as e:
            logger.warning("OSRM indisponível (%s): mapas usarão linha reta.", type(e).__name__)
            self._offline = True
            return straight
        if r.status_code == 429 or r.status_code >= 500:
            logger.warning("OSRM respondeu %s: mapas usarão linha reta.", r.status_code)
            self._offline = True
            return straight
        try:
            data = r.json()
            ok = r.status_code == 200 and data.get("code") == "Ok"
        except ValueError:
            ok = False
        if not ok:
            logger.warning("OSRM não achou trajeto (HTTP %s): usando linha reta nesta rota.", r.status_code)
            return straight
        best = data["routes"][0]
        pts = [(lat, lon) for lon, lat in best["geometry"]["coordinates"]]  # GeoJSON vem como [lon, lat]
        km = round(best["distance"] / 1000, 2)
        self._load()[key] = {"points": pts, "km": km}
        self._save()
        return RoadPath(pts, km, "osrm")


def default_router() -> RoadRouter | None:
    """Roteador configurado pelo ambiente; ``None`` quando ``ROTAS_ROADS=off``."""
    if os.getenv("ROTAS_ROADS", "osrm").strip().lower() in {"off", "0", "false", "no", "reta", "none"}:
        return None
    return RoadRouter()
