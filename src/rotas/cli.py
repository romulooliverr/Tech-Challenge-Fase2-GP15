"""Linha de comando: ``rotas <comando>`` (ou ``python -m rotas.cli <comando>``).

Comandos:
  solve          gera um cenário, roda o AG e salva solução + mapa
  experiments    experimentos do AG + comparativo de métodos (ver rotas.experiments)
  simulate-week  simula 7 dias de operação (histórico p/ relatórios semanais)
  instructions   instruções para o motorista de um veículo (LLM)
  report         relatório diário ou semanal (LLM)
  suggest        sugestões de melhoria a partir do histórico (LLM)
  chat           perguntas em linguagem natural sobre as rotas (LLM + ferramentas)
  llm-eval       avaliação da qualidade das instruções: prompt v0 (ingênuo) × v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from rotas.domain import Scenario
from rotas.evaluation import summarize
from rotas.ga.engine import GAConfig, GeneticVRP
from rotas.scenario import generate_scenario
from rotas.solution import Solution

RESULTS = Path(__file__).resolve().parents[2] / "results"
BEST = RESULTS / "best_solution.json"
COMPARISON = RESULTS / "methods_comparison.csv"
WEEK = RESULTS / "weekly_history.json"


def _assistant(solution_path: Path):
    from rotas.llm.assistant import RouteAssistant

    comp = pd.read_csv(COMPARISON) if COMPARISON.exists() else None
    week = json.loads(WEEK.read_text()) if WEEK.exists() else None
    return RouteAssistant(Solution.load(solution_path), comparison=comp, weekly_history=week)


def _write(text: str, out: Path | None) -> None:
    print(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"\n[salvo em {out}]")


def cmd_solve(a) -> None:
    sc = Scenario.load(a.scenario) if a.scenario else generate_scenario(a.deliveries, a.seed)
    cfg = GAConfig(population_size=a.population, generations=a.generations, mutation_rate=a.mutation,
                   crossover=a.crossover, local_search=a.memetic, seed=a.seed)
    res = GeneticVRP(sc, cfg).run()
    sol = Solution(sc, res.routes, f"AG ({'memético' if a.memetic else 'padrão'})", {"config": cfg.to_dict(), "tempo_s": res.elapsed_s})
    a.out.mkdir(parents=True, exist_ok=True)
    sol.save(a.out / "solution.json")
    from rotas.viz.maps import save_route_map

    save_route_map(sol, a.out / "mapa.html")
    print(json.dumps(summarize(sc, res.best), indent=2, ensure_ascii=False))
    print(f"Solução e mapa salvos em {a.out}")


def cmd_simulate_week(a) -> None:
    from rotas.simulation import save_week, simulate_week

    t0 = time.perf_counter()
    h = simulate_week(a.days, generations=a.generations)
    save_week(h, WEEK)
    print(json.dumps(h["totais"], indent=2, ensure_ascii=False))
    print(f"Histórico salvo em {WEEK} ({time.perf_counter() - t0:.0f}s)")


def cmd_instructions(a) -> None:
    asst = _assistant(a.solution)
    ids = [a.vehicle] if a.vehicle else [r["veiculo_id"] for r in asst.context["rotas"] if r["paradas"]]
    for vid in ids:
        _write(asst.driver_instructions(vid, a.prompt), a.out / f"instrucoes_veiculo_{vid}.md" if a.out else None)
        print("\n" + "=" * 80 + "\n")


def cmd_report(a) -> None:
    asst = _assistant(a.solution)
    text = asst.weekly_report() if a.kind == "semanal" else asst.daily_report()
    _write(text, a.out)


def cmd_suggest(a) -> None:
    _write(_assistant(a.solution).improvement_suggestions(), a.out)


def cmd_chat(a) -> None:
    session = _assistant(a.solution).start_chat(a.mode)
    print("Pergunte sobre as rotas do dia (linha vazia ou 'sair' para encerrar).")
    for q in a.question or []:  # perguntas passadas por argumento (modo não interativo)
        print(f"\n> {q}\n{session.send(q)}")
    if a.question:
        return
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"sair", "exit", "quit"}:
            break
        print(session.send(q))


def cmd_llm_eval(a) -> None:
    from rotas.llm.quality import evaluate_driver_text, llm_judge

    asst = _assistant(a.solution)
    rows, samples = [], {}
    for r in asst.context["rotas"]:
        if not r["paradas"]:
            continue
        route = asst.route_data(r["veiculo_id"])
        for version in ("v0", "v1"):
            text = asst.driver_instructions(r["veiculo_id"], version)
            ev = evaluate_driver_text(text, route)
            row = {
                "veiculo": r["veiculo"], "prompt": version, "geral": ev["geral"],
                "fidelidade_numerica": ev["fidelidade_numerica"]["score"],
                "nao_suportados": ", ".join(ev["fidelidade_numerica"]["nao_suportados"]),
                "cobertura": ev["cobertura_ordem"]["cobertura"],
                "ordem_correta": ev["cobertura_ordem"]["ordem_correta"],
                "destaques": ev["destaques"]["score"], "estrutura": ev["estrutura"]["score"],
                "palavras": len(text.split()),
            }
            if a.judge:
                row |= {f"juiz_{k}": v for k, v in llm_judge(asst.provider, text, route).items()}
            row["modelo"] = getattr(asst.provider, "last_model", asst.provider.name)
            rows.append(row)
            samples[f"{r['veiculo']} — {version}"] = text
            print(f"{r['veiculo']:<20} {version}: geral={ev['geral']:.3f}", flush=True)
            # salva a cada passo: uma falha de cota/rede no meio não perde o que já foi avaliado
            RESULTS.mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(RESULTS / "llm_eval.csv", index=False)
            (RESULTS / "llm_eval_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("veiculo", "prompt", "nao_suportados", "juiz_justificativa")]
    print("\nMédias por versão de prompt:\n", df.groupby("prompt")[num_cols].mean(numeric_only=True).round(3).T)
    print(f"\nDetalhes em {RESULTS / 'llm_eval.csv'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rotas", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("solve")
    p.add_argument("--scenario", type=Path)
    p.add_argument("--deliveries", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--population", type=int, default=150)
    p.add_argument("--generations", type=int, default=1500)
    p.add_argument("--mutation", type=float, default=0.6)
    p.add_argument("--crossover", default="ox", choices=["ox", "pmx", "erx"])
    p.add_argument("--memetic", action="store_true")
    p.add_argument("--out", type=Path, default=RESULTS / "solve")
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("experiments", add_help=False)
    p.set_defaults(fn=None)

    p = sub.add_parser("simulate-week")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--generations", type=int, default=800)
    p.set_defaults(fn=cmd_simulate_week)

    for name, fn in [("instructions", cmd_instructions), ("report", cmd_report), ("suggest", cmd_suggest),
                     ("chat", cmd_chat), ("llm-eval", cmd_llm_eval)]:
        p = sub.add_parser(name)
        p.add_argument("--solution", type=Path, default=BEST)
        p.set_defaults(fn=fn)
        if name == "instructions":
            p.add_argument("--vehicle", type=int)
            p.add_argument("--prompt", default="v1", choices=["v0", "v1"])
            p.add_argument("--out", type=Path)
        if name == "report":
            p.add_argument("--kind", default="diario", choices=["diario", "semanal"])
            p.add_argument("--out", type=Path)
        if name == "suggest":
            p.add_argument("--out", type=Path)
        if name == "chat":
            p.add_argument("-q", "--question", action="append", help="pergunta (pode repetir)")
            p.add_argument("--mode", choices=["context", "tools"], help="padrão: CHAT_MODE ou 'context'")
        if name == "llm-eval":
            p.add_argument("--judge", action="store_true", help="adiciona LLM-juiz (mais chamadas à API)")

    args, rest = ap.parse_known_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "experiments":
        from rotas.experiments import main as exp_main

        exp_main(rest)
        return
    if rest:
        ap.error(f"argumentos não reconhecidos: {rest}")
    args.fn(args)


if __name__ == "__main__":
    main()
