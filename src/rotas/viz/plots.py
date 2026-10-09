"""Gráficos estáticos (matplotlib) para o relatório técnico."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_convergence(histories: dict[str, list[dict]], path: Path | str, title: str = "") -> None:
    """Melhor custo por geração (uma linha por configuração; média entre seeds)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in histories.items():
        df = pd.DataFrame([{"generation": h["generation"], "best": h["best_cost"], "run": i}
                           for i, run in enumerate(runs) for h in run])
        mean = df.groupby("generation")["best"].mean()
        ax.plot(mean.index, mean.values, label=name, linewidth=1.6)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Melhor custo (km-equivalente)")
    ax.set_yscale("log")
    ax.set_title(title or "Convergência do AG (média entre seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, path: Path | str, column: str = "cost", by: str = "experiment") -> None:
    order = df.groupby(by)[column].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([df.loc[df[by] == k, column] for k in order], tick_labels=list(order))
    ax.set_ylabel(column)
    ax.set_title(f"Distribuição de {column} por configuração do AG")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_method_comparison(df: pd.DataFrame, path: Path | str) -> None:
    """Barras: custo, distância e tempo médio até críticas por método."""
    metrics = [
        ("custo_objetivo", "Custo objetivo"),
        ("distancia_total_km", "Distância (km)"),
        ("tempo_medio_ate_criticas_h", "Até as críticas (h, média)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, label) in zip(axes, metrics):
        d = df.sort_values(col)
        colors = ["#2ca02c" if v else "#d62728" for v in d["viavel"]]
        ax.barh(d["metodo"], d[col], color=colors)
        ax.set_title(label)
        if col == "custo_objetivo":
            ax.set_xscale("log")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Comparativo de métodos (verde = solução viável, vermelho = viola restrições)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
