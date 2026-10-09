# Roteiro do vídeo de demonstração (≈ 13 min)

O enunciado exige 4 coisas no vídeo: demonstração do sistema em execução, explicação dos
componentes, resultados do AG e integração com LLM. Cada bloco abaixo indica qual delas cobre.

**Preparação antes de gravar:**
- rodar `rotas experiments` e `rotas simulate-week`, para que `results/` esteja completo;
- conferir se ainda há cota do Gemini no dia (o plano gratuito renova diariamente);
- abrir `streamlit run app/streamlit_app.py`, `results/mapa_codigo_base.html` e `results/mapa_ag.html`.

| Tempo | Bloco | O que mostrar | O que falar |
|---|---|---|---|
| 0:00–1:00 | **Problema** | Slide ou README | Hospital universitário em SP, entregas para UBSs, clínicas e pacientes domiciliares. Medicamentos críticos e refrigerados, frota mista. É o "caixeiro-viajante médico", na verdade um VRP com restrições. |
| 1:00–2:30 | **Ponto de partida** (componentes) | `legacy/` e `mapa_codigo_base.html` | O código-base resolve TSP com um só veículo. Mostrar o bug `order_crossover(parent1, parent1)`. No mapa: tudo numa van, 250 kg acima da capacidade, 13 refrigerados sem refrigeração, quase 13 h de turno extra. |
| 2:30–4:30 | **Modelagem do AG** (componentes) | `docs/arquitetura.md`, diagrama do cromossomo | Permutação com separadores; OX, PMX e ERX; mutações swap, inversão (o 2-opt que era `TODO` na base) e inserção; fitness = distância + prioridade + penalidades; versão memética. |
| 4:30–6:30 | **Sistema em execução** | Streamlit → "Otimizar agora" | Ajustar entregas e população, clicar em Otimizar e mostrar a convergência ao vivo. Abrir o mapa: cores por veículo, críticas em vermelho, popups. Mostrar a tabela da frota. |
| 6:30–8:30 | **Resultados do AG** | Aba Comparativo | 9 experimentos × 5 seeds. Achados: mutação alta ajuda; semear com o vizinho mais próximo causa convergência prematura; ERX é ruim com separadores; o memético empata com o OR-Tools (449,6 contra 449,1). Escalabilidade: 20, 40 e 80 entregas. |
| 8:30–10:00 | **LLM: instruções** | Aba Instruções → Van Refrigerada 2 | Seções fixas, ordem de carregamento invertida, marcadores 🔴 ❄️ ⏰ e alerta de janela apertada. Explicar o grounding: os dados vão em `<dados>` e as contas são feitas em Python. |
| 10:00–11:30 | **LLM: relatórios e sugestões** | Aba Relatórios → Semanal e Sugestões | 24,4% menos km, R$ 799 de economia, 27 h a menos de condução, críticas em 1,8 h contra 4,7 h. Sugestões com evidência citada dos dados. |
| 11:30–12:30 | **LLM: chat** | Aba "Pergunte às rotas" | Perguntar: "Quais entregas críticas a Van Refrigerada 2 faz?", "E a Moto 5?" (usa o histórico), "Quem leva insulina para a Vila Prudente?" e uma pergunta fora de escopo. |
| 12:30–13:15 | **Qualidade da LLM** | `results/llm_eval.csv` | Prompt v0 contra v1: nota 0,82 → 1,00; o v0 inventou números. Mostrar o achado do chat (JSON longo fez o modelo lite errar; tabelas resolveram) e a troca automática de modelo quando a cota acaba. |
| 13:15–13:45 | **Encerramento** | README | Testes (`pytest`: 40 testes), limitações e próximos passos (distâncias reais por OSRM, múltiplas viagens por veículo, integração com o Módulo 3). |
