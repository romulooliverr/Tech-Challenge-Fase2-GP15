# Arquitetura

## Visão geral

```mermaid
flowchart LR
    subgraph Dados
        SC[Gerador de cenários<br/>São Paulo · seed] --> SCN[(Cenário<br/>entregas · frota · turno)]
        SCN --> DM[Matriz de distâncias<br/>Haversine × 1,3]
    end

    subgraph Otimização
        GA[Algoritmo Genético<br/>OX/PMX/ERX · swap/inversão/inserção<br/>torneio · elitismo]
        LS[Busca local<br/>relocate + 2-opt]
        BL[Baselines<br/>aleatório · vizinho mais próximo<br/>Clarke-Wright · OR-Tools · código-base]
        EV{{Função de custo única<br/>distância + prioridade + penalidades}}
        GA <--> LS
        GA --> EV
        BL --> EV
    end

    subgraph Saídas
        SOL[(Solução<br/>rotas por veículo)]
        MAP[Mapa Folium]
        PLT[Gráficos e CSVs]
        SIM[Simulação semanal<br/>histórico + indicadores]
    end

    subgraph LLM["LLM — Google Gemini"]
        CTX[Contexto factual<br/>route_details · indicadores<br/>pré-calculados]
        PR[Prompts<br/>persona · grounding · formato]
        INS[Instruções aos motoristas]
        REP[Relatórios diário/semanal]
        SUG[Sugestões de melhoria]
        CHAT[Chat: modo contexto<br/>ou function calling]
        QA[Avaliação de qualidade<br/>fidelidade · cobertura · ordem]
    end

    DM --> GA & BL
    EV --> SOL --> MAP & PLT
    SOL --> CTX
    SIM --> CTX
    CTX --> PR --> INS & REP & SUG & CHAT
    INS --> QA

    UI[Streamlit] -.-> GA & MAP & INS & REP & CHAT
    CLI[CLI rotas] -.-> GA & SIM & INS & REP & CHAT & QA
```

## Representação genética

```mermaid
flowchart LR
    C["Cromossomo: [12, 3, 7, <b>41</b>, 9, 1, <b>42</b>, 5, 2, 8, <b>43</b>, <b>44</b>, 4, 6]<br/>1..40 = entregas · 41..44 = separadores"]
    C --> V1["Van Refrigerada 1: 12 → 3 → 7"]
    C --> V2["Van Refrigerada 2: 9 → 1"]
    C --> V3["Van Carga 3: 5 → 2 → 8"]
    C --> V4["Moto 4: (vazia)"]
    C --> V5["Moto 5: 4 → 6"]
```

Cada rota é simulada no tempo, do depósito às 08:00 até o retorno. São acumulados distância,
carga, horário de chegada (com espera, se o veículo chegar antes da janela abrir) e atrasos.

## Laço evolutivo

```mermaid
flowchart TD
    A[População inicial aleatória] --> B[Avaliar custo<br/>cache por cromossomo]
    B --> C{AG memético e<br/>geração múltipla de 10?}
    C -- sim --> D[relocate + 2-opt<br/>nos 2 melhores]
    C -- não --> E
    D --> E[Elitismo: copia os 2 melhores]
    E --> F[Seleção por torneio k=3]
    F --> G[Cruzamento OX p=0,9]
    G --> H[Mutação p=0,6<br/>swap | inversão | inserção]
    H --> I{Gerações<br/>esgotadas?}
    I -- não --> B
    I -- sim --> J[Melhor solução]
```

## Fluxo da LLM

```mermaid
sequenceDiagram
    participant U as Usuário / Motorista / Gestor
    participant A as RouteAssistant
    participant P as Python (fatos e contas)
    participant G as Gemini
    U->>A: "Gere as instruções da Van 2"
    A->>P: route_details(solução)
    P-->>A: paradas, horários, itens, flags
    A->>G: system (persona + regras) + prompt v1 + <dados>JSON</dados>
    G-->>A: Markdown com seções fixas
    A->>P: quality.evaluate_driver_text (opcional)
    A-->>U: instruções
    Note over A,G: Cota diária esgotada → próximo modelo da cadeia<br/>429 por minuto → espera exponencial
```

## Componentes e responsabilidades

| Módulo | Responsabilidade |
|---|---|
| `domain.py`, `geo.py`, `scenario.py` | Modelo do problema e geração de instâncias reprodutíveis |
| `evaluation.py` | Decodificação, simulação no tempo, custo e restrições. É a **fonte única de verdade** |
| `ga/` | Operadores, laço evolutivo, busca local |
| `baselines/` | Métodos de comparação, todos reavaliados pela mesma função de custo |
| `solution.py` | Persistência e descrição detalhada das rotas, que é o contexto factual da LLM |
| `simulation.py` | Semana simulada e indicadores agregados |
| `llm/` | Provedores, prompts, assistente e avaliação de qualidade |
| `viz/` | Mapa interativo e gráficos |
| `cli.py`, `app/` | Interfaces |

## Observabilidade

- Logs com `logging`: o AG registra o progresso a cada 25 gerações, e o provedor de LLM
  registra novas tentativas, troca de modelo e erros de cota.
- `results/run_info.json` guarda os parâmetros de cada rodada de experimentos.
- A avaliação de LLM registra qual modelo gerou cada texto (coluna `modelo`).
