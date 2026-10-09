"""Gerador de cenários sintéticos (e reprodutíveis) para a cidade de São Paulo.

As coordenadas partem do centro aproximado de distritos reais, com um deslocamento
aleatório de até ~1,5 km. Os nomes das unidades são genéricos/fictícios.
"""

from __future__ import annotations

import numpy as np

from rotas.domain import Delivery, Location, Priority, Scenario, Vehicle

DEPOT = Location(
    name="CD Farmacêutico Central — Hospital Universitário",
    lat=-23.5570,
    lon=-46.6700,
    kind="deposito",
    district="Pinheiros",
)

# Centro aproximado (lat, lon) de distritos de São Paulo.
DISTRICTS: dict[str, tuple[float, float]] = {
    "Santana": (-23.5017, -46.6250),
    "Tucuruvi": (-23.4800, -46.6030),
    "Vila Maria": (-23.5120, -46.5800),
    "Casa Verde": (-23.5100, -46.6560),
    "Freguesia do Ó": (-23.5000, -46.6960),
    "Brasilândia": (-23.4700, -46.6900),
    "Pirituba": (-23.4870, -46.7240),
    "Lapa": (-23.5270, -46.7040),
    "Perdizes": (-23.5360, -46.6770),
    "Butantã": (-23.5710, -46.7080),
    "Sé": (-23.5500, -46.6340),
    "Mooca": (-23.5590, -46.5990),
    "Tatuapé": (-23.5400, -46.5760),
    "Penha": (-23.5260, -46.5440),
    "Vila Prudente": (-23.5830, -46.5800),
    "Ipiranga": (-23.5890, -46.6060),
    "Vila Mariana": (-23.5890, -46.6340),
    "Jabaquara": (-23.6460, -46.6410),
    "Santo Amaro": (-23.6540, -46.7100),
    "Cidade Ademar": (-23.6740, -46.6540),
    "Campo Limpo": (-23.6450, -46.7600),
    "Capão Redondo": (-23.6720, -46.7790),
    "Jardim Ângela": (-23.7110, -46.7720),
    "Itaquera": (-23.5400, -46.4560),
    "São Miguel Paulista": (-23.4980, -46.4440),
    "Guaianases": (-23.5410, -46.4120),
    "São Mateus": (-23.6040, -46.4770),
}

# tipo de unidade: (peso no sorteio, faixa de demanda kg, tempo de serviço min, prefixo do nome)
UNIT_TYPES = {
    "hospital": (0.10, (40, 110), 15, "Hospital Dia"),
    "ubs": (0.40, (10, 45), 10, "UBS"),
    "clinica": (0.20, (5, 25), 8, "Clínica de Especialidades"),
    "domiciliar": (0.30, (1, 6), 12, "Atendimento Domiciliar"),
}

ITEMS = {
    "refrigerado": ["Insulina", "Vacinas", "Imunobiológicos", "Hemoderivados"],
    "critico": [
        "Antibióticos de amplo espectro",
        "Anticoagulantes",
        "Quimioterápicos",
        "Sedativos de UTI",
        "Drogas vasoativas",
    ],
    "regular": [
        "Luvas e EPIs",
        "Gaze e curativos",
        "Seringas e agulhas",
        "Soro fisiológico",
        "Analgésicos",
        "Material de higiene",
    ],
}

PRIORITY_PROBS = {Priority.CRITICA: 0.2, Priority.URGENTE: 0.3, Priority.REGULAR: 0.5}


def default_fleet() -> list[Vehicle]:
    """Frota mista: vans refrigeradas, van comum e motos (rápidas, pouca carga)."""
    return [
        Vehicle(1, "Van Refrigerada 1", "van", 400, 180, 22, True, 1.8),
        Vehicle(2, "Van Refrigerada 2", "van", 400, 180, 22, True, 1.8),
        Vehicle(3, "Van Carga 3", "van", 600, 200, 22, False, 1.5),
        Vehicle(4, "Moto 4", "moto", 25, 120, 32, False, 0.5),
        Vehicle(5, "Moto 5", "moto", 25, 120, 32, False, 0.5),
    ]


def generate_scenario(
    n_deliveries: int = 40,
    seed: int = 42,
    vehicles: list[Vehicle] | None = None,
    name: str | None = None,
) -> Scenario:
    rng = np.random.default_rng(seed)
    district_names = list(DISTRICTS)
    types = list(UNIT_TYPES)
    type_probs = np.array([UNIT_TYPES[t][0] for t in types])
    counters: dict[str, int] = {}
    deliveries: list[Delivery] = []

    for i in range(1, n_deliveries + 1):
        kind = types[rng.choice(len(types), p=type_probs / type_probs.sum())]
        _, (dmin, dmax), service, prefix = UNIT_TYPES[kind]
        district = district_names[rng.integers(len(district_names))]
        clat, clon = DISTRICTS[district]
        # ~0,009° ≈ 1 km
        lat, lon = clat + rng.normal(0, 0.007), clon + rng.normal(0, 0.007)
        counters[f"{kind}-{district}"] = counters.get(f"{kind}-{district}", 0) + 1
        unit_name = (
            f"Paciente domiciliar #{i:02d} ({district})"
            if kind == "domiciliar"
            else f"{prefix} {district} {counters[f'{kind}-{district}']:02d}"
        )

        priority = Priority(
            rng.choice([int(p) for p in PRIORITY_PROBS], p=list(PRIORITY_PROBS.values()))
        )
        refrigerated = bool(rng.random() < (0.45 if kind == "domiciliar" else 0.2))
        items: list[str] = []
        if refrigerated:
            items.append(ITEMS["refrigerado"][rng.integers(len(ITEMS["refrigerado"]))])
        if priority == Priority.CRITICA:
            items.append(ITEMS["critico"][rng.integers(len(ITEMS["critico"]))])
        n_regular = int(rng.integers(0 if items else 1, 3))
        items += [str(x) for x in rng.choice(ITEMS["regular"], size=n_regular, replace=False)]

        window = None
        if kind == "domiciliar":
            start = int(rng.integers(8, 14)) * 60 + int(rng.choice([0, 30]))
            window = (start, start + int(rng.choice([120, 180])))

        deliveries.append(
            Delivery(
                id=i,
                location=Location(unit_name, round(lat, 6), round(lon, 6), kind, district),
                demand_kg=round(float(rng.uniform(dmin, dmax)), 1),
                priority=priority,
                refrigerated=refrigerated,
                service_min=float(service),
                items=tuple(dict.fromkeys(items)),
                window=window,
            )
        )

    return Scenario(
        depot=DEPOT,
        deliveries=deliveries,
        vehicles=vehicles or default_fleet(),
        name=name or f"sp_{n_deliveries}_seed{seed}",
    )
