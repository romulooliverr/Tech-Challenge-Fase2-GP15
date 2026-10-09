"""Gera notebooks/demo.ipynb (executar depois com nbconvert)."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Tech Challenge Fase 2 — Projeto 2: Rotas de medicamentos com AG + LLM\n\n"
       "Notebook de demonstração. Para rodar tudo do zero: `rotas experiments`, `rotas simulate-week`.\n"
       "Aqui usamos os resultados já salvos em `results/` e fazemos uma otimização ao vivo, menor.\n\n"
       "> As chamadas à LLM usam `LLM_PROVIDER` do `.env`. Para não gastar cota, o notebook exibe os textos "
       "já gerados pelo Gemini (em `results/llm/`) e faz **uma** chamada ao vivo opcional no final."),
    code("import json, os\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import Markdown, Image, display\n"
         "os.chdir(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd())\nRES = Path('results')"),
    md("## 1. Cenário: São Paulo, 40 entregas, frota mista"),
    code("from rotas.domain import Scenario\nsc = Scenario.load(RES / 'scenario.json')\n"
         "pd.DataFrame([{ 'id': d.id, 'destino': d.location.name, 'tipo': d.location.kind, 'prioridade': d.priority.label,\n"
         "   'refrigerado': d.refrigerated, 'kg': d.demand_kg, 'janela': d.window} for d in sc.deliveries]).head(12)"),
    code("pd.DataFrame([v.__dict__ for v in sc.vehicles])"),
    md("## 2. Algoritmo genético ao vivo (300 gerações, ~2 s)"),
    code("from rotas.ga.engine import GeneticVRP, GAConfig\nfrom rotas.evaluation import summarize\n"
         "res = GeneticVRP(sc, GAConfig(generations=300, local_search=True, seed=7)).run()\n"
         "summarize(sc, res.best)"),
    code("import matplotlib.pyplot as plt\nh = pd.DataFrame([x.__dict__ for x in res.history])\n"
         "ax = h.plot(x='generation', y=['best_cost', 'mean_cost'], logy=True, figsize=(8, 4), title='Convergência')\nplt.show()"),
    md("## 3. Experimentos (9 configurações × 5 seeds, 1500 gerações)"),
    code("pd.read_csv(RES / 'experiments_summary.csv')"),
    code("display(Image(RES / 'convergence.png'))\ndisplay(Image(RES / 'boxplot_cost.png'))"),
    md("## 4. Comparativo com outras abordagens"),
    code("pd.read_csv(RES / 'methods_comparison.csv')[['metodo','custo_objetivo','distancia_total_km','viavel','tempo_medio_ate_criticas_h','veiculos_usados','tempo_s']]"),
    code("display(Image(RES / 'methods_comparison.png'))\n"
         "if (RES / 'scalability.csv').exists():\n    display(pd.read_csv(RES / 'scalability.csv').pivot(index='metodo', columns='entregas', values='custo'))"),
    md("## 5. Mapa da melhor solução"),
    code("from rotas.solution import Solution\nfrom rotas.viz.maps import route_map\nbest = Solution.load(RES / 'best_solution.json')\nroute_map(best)"),
    md("## 6. Semana simulada (AG × método anterior)"),
    code("week = json.loads((RES / 'weekly_history.json').read_text())\npd.Series(week['totais'])"),
    md("## 7. LLM — textos gerados pelo Gemini"),
    code("display(Markdown((RES / 'llm' / 'instrucoes_van_refrigerada_2_v1.md').read_text()))"),
    code("display(Markdown((RES / 'llm' / 'relatorio_semanal.md').read_text()))"),
    code("display(Markdown((RES / 'llm' / 'sugestoes.md').read_text()))"),
    md("### Avaliação de qualidade: prompt ingênuo (v0) × prompt elaborado (v1)"),
    code("ev = pd.read_csv(RES / 'llm_eval.csv')\nev.groupby('prompt')[['geral','fidelidade_numerica','cobertura','destaques','estrutura','palavras']].mean().round(3)"),
    md("## 8. Chat (uma pergunta ao vivo — consome 1 requisição da cota)"),
    code("from rotas.llm.assistant import RouteAssistant\n"
         "asst = RouteAssistant(best, comparison=pd.read_csv(RES / 'methods_comparison.csv'))\n"
         "chat = asst.start_chat()\nprint(chat.send('Quais entregas críticas a Van Refrigerada 2 faz e em que horário?'))"),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
Path(__file__).resolve().parents[1].joinpath("notebooks", "demo.ipynb").write_text(nbf.writes(nb))
print("notebooks/demo.ipynb gerado")
