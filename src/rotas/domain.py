"""Modelo de domínio: depósito, entregas, veículos e cenário (instância do problema)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from rotas.geo import distance_matrix


class Priority(IntEnum):
    REGULAR = 1  # insumos de reposição (luvas, gaze, seringas...)
    URGENTE = 2  # medicamentos com estoque baixo na unidade
    CRITICA = 3  # medicamentos críticos (UTI, quimioterápicos, anticoagulantes...)

    @property
    def label(self) -> str:
        return {1: "regular", 2: "urgente", 3: "crítica"}[int(self)]


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    kind: str  # deposito | hospital | ubs | clinica | domiciliar
    district: str


@dataclass(frozen=True)
class Delivery:
    """Uma parada de entrega. ``id`` vai de 1 a n (0 é reservado ao depósito)."""

    id: int
    location: Location
    demand_kg: float
    priority: Priority
    refrigerated: bool  # exige cadeia fria (vacinas, insulina...)
    service_min: float  # tempo de descarga/conferência na parada
    items: tuple[str, ...] = ()
    window: tuple[int, int] | None = None  # janela de atendimento (min desde 00:00)


@dataclass(frozen=True)
class Vehicle:
    id: int
    name: str
    kind: str  # van | moto
    capacity_kg: float
    max_range_km: float  # autonomia (combustível/bateria) por turno
    speed_kmh: float  # velocidade média urbana
    refrigerated: bool
    cost_per_km: float  # R$/km (combustível + desgaste), usado nos relatórios


@dataclass
class Scenario:
    depot: Location
    deliveries: list[Delivery]
    vehicles: list[Vehicle]
    shift_start_min: int = 8 * 60
    shift_end_min: int = 17 * 60
    road_factor: float = 1.3
    name: str = "cenario"
    dist: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ids = [d.id for d in self.deliveries]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError("IDs das entregas devem ser 1..n, em ordem.")
        points = [(self.depot.lat, self.depot.lon)] + [
            (d.location.lat, d.location.lon) for d in self.deliveries
        ]
        self.dist = distance_matrix(points, self.road_factor)

    @property
    def n_deliveries(self) -> int:
        return len(self.deliveries)

    @property
    def n_vehicles(self) -> int:
        return len(self.vehicles)

    def delivery(self, delivery_id: int) -> Delivery:
        return self.deliveries[delivery_id - 1]

    # ------------------------------------------------------------------ serialização
    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "shift_start_min": self.shift_start_min,
            "shift_end_min": self.shift_end_min,
            "road_factor": self.road_factor,
            "depot": asdict(self.depot),
            "vehicles": [asdict(v) for v in self.vehicles],
            "deliveries": [],
        }
        for dl in self.deliveries:
            row = asdict(dl)
            row["priority"] = int(dl.priority)
            d["deliveries"].append(row)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        deliveries = [
            Delivery(
                id=r["id"],
                location=Location(**r["location"]),
                demand_kg=r["demand_kg"],
                priority=Priority(r["priority"]),
                refrigerated=r["refrigerated"],
                service_min=r["service_min"],
                items=tuple(r.get("items", ())),
                window=tuple(r["window"]) if r.get("window") else None,
            )
            for r in d["deliveries"]
        ]
        return cls(
            depot=Location(**d["depot"]),
            deliveries=deliveries,
            vehicles=[Vehicle(**v) for v in d["vehicles"]],
            shift_start_min=d["shift_start_min"],
            shift_end_min=d["shift_end_min"],
            road_factor=d["road_factor"],
            name=d.get("name", "cenario"),
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path | str) -> Scenario:
        return cls.from_dict(json.loads(Path(path).read_text()))


def fmt_time(minutes: float) -> str:
    """480 -> '08:00'."""
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"
