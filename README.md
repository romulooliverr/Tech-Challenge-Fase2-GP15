# Tech Challenge — Fase 2 · Projeto 2
## Otimização de Rotas para Distribuição de Medicamentos e Insumos (VRP) com Algoritmo Genético + LLM

**FIAP — Pós Tech IA para Devs (IADT)**

Sistema que planeja as rotas diárias de entrega de medicamentos e insumos de um hospital
universitário em **São Paulo** para UBSs, clínicas, hospitais-dia e pacientes em atendimento
domiciliar. Ele combina:

1. um **Algoritmo Genético (AG)** para o *problema de roteamento de veículos* (VRP), evoluído a
   partir do código-base de TSP fornecido. O AG considera prioridades, capacidade, autonomia,
   frota mista, cadeia fria, janelas de horário e fim de turno;
2. uma **LLM (Google Gemini)** que gera instruções para os motoristas, relatórios
   diários e semanais e sugestões de melhoria, e responde perguntas em linguagem natural
   sobre as rotas.

> 📄 Relatório técnico completo: [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) ·
> Arquitetura: [`docs/arquitetura.md`](docs/arquitetura.md) ·
> Roteiro do vídeo: [`docs/roteiro_video.md`](docs/roteiro_video.md)

---

## Resultados principais

Cenário padrão: 40 entregas, 850 kg, 9 críticas, 13 refrigeradas, 17 com janela de horário,
e frota de 5 veículos (2 vans refrigeradas, 1 van de carga e 2 motos).

| Método | Custo objetivo | Distância | Viável? | Tempo até críticas |
|---|---:|---:|:---:|---:|
| **AG memético** (melhor de 5 seeds) | **449,6** | **264,1 km** | ✅ | **2,17 h** |
| OR-Tools (busca local guiada, 30 s) | 449,1 | 264,6 km | ✅ | 2,20 h |
| AG padrão (melhor seed) | 463,8 | 273,7 km | ✅ | 2,31 h |
| Clarke-Wright (savings) | 979,1 | 314,1 km | ❌ | 3,14 h |
| Vizinho mais próximo | 1.659,7 | 371,6 km | ❌ | 4,67 h |
| Código-base original (TSP, 1 veículo) | 87.668,8 | 311,1 km | ❌ | 13,91 h |

**Simulação de uma semana** (7 dias, 284 entregas), comparando o AG com o método anterior
(vizinho mais próximo):

- **−24,4% de distância** (692 km a menos)
- **−R$ 798,73** de custo operacional
- **−27,4 h** de condução
- entregas críticas em **1,79 h** em média, contra 4,72 h
- **7 de 7 dias** com todas as restrições atendidas, contra 2 de 7

**Engenharia de prompt** (avaliação automática das instruções para motoristas): a nota geral
subiu de **0,82** com o prompt ingênuo para **1,00** com o prompt elaborado. O prompt
ingênuo inventou números, deixou paradas de fora e não seguiu uma estrutura.

---

## Estrutura do repositório

```
├── src/rotas/
│   ├── domain.py            # entidades: entrega, veículo, cenário
│   ├── geo.py               # Haversine + fator de circuidade urbana
│   ├── scenario.py          # gerador de cenários de São Paulo (reprodutível por seed)
│   ├── evaluation.py        # decodificação do cromossomo + função de custo (fitness)
│   ├── ga/
│   │   ├── operators.py     # seleção, cruzamento (OX, PMX, ERX) e mutação (swap, inversão, inserção)
│   │   ├── local_search.py  # relocate + 2-opt (AG memético)
│   │   └── engine.py        # laço evolutivo
│   ├── baselines/           # aleatório, vizinho mais próximo, Clarke-Wright, OR-Tools, código-base
│   ├── viz/                 # mapa Folium + gráficos matplotlib
│   ├── llm/
│   │   ├── provider.py      # Gemini (com modelos de reserva) + Mock offline
│   │   ├── prompts.py       # prompts documentados (técnicas de prompt engineering)
│   │   ├── assistant.py     # instruções, relatórios, sugestões, chat (contexto ou ferramentas)
│   │   └── quality.py       # avaliação automática da qualidade dos textos
│   ├── simulation.py        # simulação semanal (histórico para relatórios)
│   ├── experiments.py       # experimentos do AG + comparativo de métodos
│   └── cli.py               # linha de comando `rotas`
├── app/streamlit_app.py     # interface web de demonstração
├── legacy/                  # código-base de TSP fornecido (com 4 bugs corrigidos e anotados)
├── tests/                   # 53 testes automatizados (pytest)
├── scripts/scalability.py   # estudo de escalabilidade (20/40/80 entregas)
├── notebooks/demo.ipynb     # notebook de demonstração
├── results/                 # resultados, gráficos, mapas e textos gerados pela LLM
└── docs/                    # relatório técnico, arquitetura, roteiro do vídeo
```

---

## Como executar

### 1. Ambiente

Requer Python 3.11 a 3.13. Funciona com **Poetry** ou com **venv + pip**:

```bash
# Opção A — Poetry
poetry install --extras dev
poetry shell

# Opção B — venv + pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Chave do Gemini (gratuita)

```bash
cp .env.example .env
# edite .env e coloque GEMINI_API_KEY=... (gere em https://aistudio.google.com/apikey)
```

O **plano gratuito tem cota diária pequena por modelo**: no `gemini-3.5-flash` foram só 20
requisições por dia. Quando a cota de um modelo acaba, o sistema passa automaticamente para os
modelos de reserva (`GEMINI_FALLBACK_MODELS`, por padrão `gemini-3.5-flash-lite` e
`gemini-3.1-flash-lite`). Sem chave, ou com `LLM_PROVIDER=mock`, tudo roda offline com
respostas determinísticas.

**Alternativa local (Ollama), sem cota e sem internet:**

```bash
ollama serve &                 # em outro terminal, se o app do Ollama não estiver aberto
ollama pull qwen2.5:14b        # ou outro modelo; ajuste OLLAMA_MODEL no .env
# no .env: LLM_PROVIDER=ollama  (OLLAMA_BASE_URL padrão: http://localhost:11434)
```

### 3. Comandos

```bash
rotas experiments              # 9 configurações × 5 seeds + comparativo (~4 min)
rotas experiments --quick      # versão rápida (~30 s)
rotas simulate-week            # 7 dias de operação simulados (~15 s)
rotas solve --memetic          # gera um cenário, otimiza e salva solução + mapa
rotas map --all                # regenera os mapas a partir das soluções salvas (sem reotimizar)

rotas instructions --vehicle 1 # instruções para o motorista (LLM)
rotas report --kind diario     # relatório diário (LLM)
rotas report --kind semanal    # relatório semanal (LLM)
rotas suggest                  # sugestões de melhoria (LLM)
rotas chat                     # perguntas em linguagem natural (LLM)
rotas chat -q "Quem leva insulina para a Vila Prudente?"
rotas llm-eval                 # avaliação de qualidade: prompt v0 × v1

streamlit run app/streamlit_app.py   # interface web
pytest                               # testes
```

Os mapas interativos ficam em `results/mapa_ag.html`, `results/mapa_vizinho_mais_proximo.html`
e `results/mapa_codigo_base.html`. Para vê-los, basta abrir no navegador.
O trajeto de cada veículo é desenhado pelas ruas (OSRM público, sem chave; cache em
`results/road_cache.json`). Sem internet, ou com `ROTAS_ROADS=off`, o mapa usa linhas retas
tracejadas. O km usado na otimização continua sendo Haversine × 1,3.

---

## Decisões de projeto, em resumo

| Decisão | Por quê |
|---|---|
| Permutação com **genes separadores** | Os operadores de permutação do código-base (OX) continuam válidos, e o próprio AG decide qual entrega vai em qual veículo. |
| **Restrições suaves** (penalidade proporcional) | Permite ao AG atravessar regiões inviáveis rumo a soluções viáveis. Todas as soluções finais relevantes são viáveis. |
| **Mesma função de custo para todos os métodos** | Torna a comparação justa: o resultado do OR-Tools também é reavaliado por ela. |
| **AG memético** (relocate + 2-opt) | Fechou a diferença para o OR-Tools (0,1%). |
| **Sem contas na LLM** | Economias e percentuais são calculados em Python, e a LLM só interpreta. Isso eliminou números inventados. |
| **Chat em modo contexto** (padrão) ou **ferramentas** | O modo contexto usa 1 requisição por pergunta, o que cabe no plano gratuito. O modo ferramentas (*function calling*) escala para bases maiores. |

Detalhes, experimentos e discussão estão em [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md).
