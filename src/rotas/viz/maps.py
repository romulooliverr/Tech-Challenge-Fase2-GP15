"""Mapa interativo (Folium/Leaflet) das rotas otimizadas."""

from __future__ import annotations

from pathlib import Path

import folium

from rotas.domain import Priority
from rotas.solution import Solution, route_details

VEHICLE_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]
PRIORITY_COLORS = {"crítica": "red", "urgente": "orange", "regular": "green"}


def route_map(solution: Solution, title: str | None = None) -> folium.Map:
    sc = solution.scenario
    details = route_details(sc, solution.evaluate())
    m = folium.Map(location=[sc.depot.lat, sc.depot.lon], zoom_start=11, tiles="OpenStreetMap")

    folium.Marker(
        [sc.depot.lat, sc.depot.lon],
        tooltip=sc.depot.name,
        icon=folium.Icon(color="black", icon="hospital-o", prefix="fa"),
    ).add_to(m)

    for j, route in enumerate(details):
        if not route["usado"]:
            continue
        color = VEHICLE_COLORS[j % len(VEHICLE_COLORS)]
        layer = folium.FeatureGroup(
            name=f"{route['veiculo']} — {route['distancia_km']} km, {len(route['paradas'])} paradas"
        )
        path = [(sc.depot.lat, sc.depot.lon)] + [(s["lat"], s["lon"]) for s in route["paradas"]]
        path.append((sc.depot.lat, sc.depot.lon))
        folium.PolyLine(path, color=color, weight=4, opacity=0.8).add_to(layer)
        for s in route["paradas"]:
            popup = (
                f"<b>{s['ordem']}. {s['destino']}</b><br>"
                f"Veículo: {route['veiculo']}<br>"
                f"Prioridade: {s['prioridade']}{' · ❄️ refrigerado' if s['refrigerado'] else ''}<br>"
                f"Carga: {s['carga_kg']} kg<br>Itens: {', '.join(s['itens'])}<br>"
                f"Chegada prevista: {s['chegada_prevista']}"
                + (f"<br>Janela: {s['janela']}" if s["janela"] else "")
            )
            folium.CircleMarker(
                [s["lat"], s["lon"]],
                radius=9 if s["prioridade"] == Priority.CRITICA.label else 6,
                color=color,
                weight=3,
                fill=True,
                fill_color=PRIORITY_COLORS[s["prioridade"]],
                fill_opacity=0.9,
                popup=folium.Popup(popup, max_width=320),
                tooltip=f"{route['veiculo']} · parada {s['ordem']}",
            ).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    # enquadra o mapa no depósito + todas as entregas (em vez de um zoom fixo)
    lats = [sc.depot.lat] + [d.location.lat for d in sc.deliveries]
    lons = [sc.depot.lon] + [d.location.lon for d in sc.deliveries]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    legend = (
        "<div style='position:fixed;bottom:24px;left:24px;z-index:9999;background:white;"
        "padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font-size:13px;"
        "color:#222;line-height:1.5'>"
        f"<b>{title or solution.method}</b><br>"
        "<span style='color:red'>●</span> crítica &nbsp;"
        "<span style='color:orange'>●</span> urgente &nbsp;"
        "<span style='color:green'>●</span> regular<br>"
        "Cor da linha = veículo · marcador grande = entrega crítica</div>"
    )
    m.get_root().html.add_child(folium.Element(legend))
    return m


def save_route_map(solution: Solution, path: Path | str, title: str | None = None) -> Path:
    path = Path(path)
    route_map(solution, title).save(str(path))
    return path
