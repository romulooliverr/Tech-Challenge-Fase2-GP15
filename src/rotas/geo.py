"""Distâncias geográficas.

Usamos Haversine (distância em linha reta sobre a esfera terrestre) multiplicada por um
*fator de circuidade* (~1,3 em malhas urbanas como São Paulo) para aproximar a distância
percorrida pelas ruas, sem depender de APIs externas de roteamento.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def distance_matrix(points: Sequence[tuple[float, float]], road_factor: float = 1.3) -> np.ndarray:
    """Matriz simétrica (n x n) de distâncias em km por vias aproximadas."""
    lat = np.radians(np.array([p[0] for p in points]))[:, None]
    lon = np.radians(np.array([p[1] for p in points]))[:, None]
    dlat, dlon = lat - lat.T, lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))) * road_factor
