"""Prompts (engenharia de prompt documentada).

Técnicas aplicadas — ver docs/relatorio_tecnico.md, seção LLM:

1. **Papel/persona** no system prompt (coordenador de logística hospitalar).
2. **Ancoragem (grounding)**: todos os fatos vão num bloco ``<dados>`` em JSON e o modelo é
   instruído a usar SOMENTE esses dados e a dizer "não consta nos dados" quando faltar algo.
3. **Sem aritmética no LLM**: percentuais, economias e totais são calculados em Python e
   entregues prontos (campo ``indicadores``) — LLMs erram contas com frequência.
4. **Formato de saída explícito**: seções fixas em Markdown → previsível e avaliável.
5. **Adequação ao público**: motorista (frases curtas, imperativas) × gestor (executivo).
6. **Few-shot**: exemplo do formato de uma parada.
7. **Prompt "ingênuo" (v0)** mantido para comparação na avaliação de qualidade.
"""

from __future__ import annotations

import json

SYSTEM_BASE = """Você é o assistente de logística da Central de Distribuição Farmacêutica de um \
hospital universitário em São Paulo, responsável por entregar medicamentos e insumos para \
unidades de saúde e pacientes em atendimento domiciliar.

Regras obrigatórias:
- Use SOMENTE as informações dentro de <dados>. Não invente endereços, nomes, horários, \
quantidades nem números. Se algo não estiver nos dados, escreva "não consta nos dados".
- Não faça contas: use os valores já calculados que estão nos dados.
- Escreva em português do Brasil.
- Entregas de prioridade "crítica" (ex.: quimioterápicos, drogas de UTI) e itens refrigerados \
(cadeia fria, 2 °C a 8 °C) devem sempre ser destacados.
- Você apoia a operação; decisões clínicas não são sua responsabilidade."""

# ------------------------------------------------------------------ instruções ao motorista

DRIVER_V0 = "Escreva instruções para o motorista com base nestes dados:\n<dados>{dados}</dados>"

DRIVER_V1 = """Gere as instruções de trabalho do dia para o motorista do veículo abaixo.
Público: motorista e auxiliar de entrega — frases curtas, imperativas, sem jargão técnico.

<dados>{dados}</dados>

Responda EXATAMENTE com estas seções, nesta ordem:

## Resumo da rota
Veículo, horário de saída, retorno previsto, distância total, carga total e número de paradas \
(diga quantas são críticas e quantas são refrigeradas).

## Antes de sair
Checklist de carregamento. Carregue na ordem INVERSA das entregas (a última entrega vai ao \
fundo do veículo). Se houver refrigerados, inclua a conferência da temperatura (2 °C a 8 °C).

## Paradas
Uma entrada por parada, NA ORDEM dos dados, seguindo exatamente este formato:

Exemplo (formato apenas — não copie os valores):
**3. UBS Exemplo 01** (Bairro) — chegada prevista 09:40
- Entregar: 22.5 kg — Seringas e agulhas, Soro fisiológico
- 🔴 CRÍTICA · ❄️ REFRIGERADO · ⏰ janela 10:00-12:00 (omita os marcadores que não se aplicam)
- Conferir itens com o responsável da unidade e colher assinatura.

## Atenção
Somente riscos presentes nos dados: janelas apertadas, esperas, entregas críticas no fim da \
rota, uso alto da autonomia ou da capacidade.

## Em caso de imprevisto
Orientações curtas: avisar a Central, priorizar entregas críticas e refrigeradas, nunca deixar \
refrigerados fora da caixa térmica."""

# ------------------------------------------------------------------ relatórios

DAILY_REPORT = """Escreva o relatório DIÁRIO de eficiência das rotas para a coordenação de \
logística e a diretoria do hospital. Público: gestores — linguagem executiva e objetiva.

<dados>{dados}</dados>

Seções obrigatórias:
## Resumo executivo
Três tópicos com os resultados mais importantes do dia.
## Indicadores do dia
Tabela Markdown com distância, custo operacional (R$), veículos usados, viabilidade e tempo \
médio até as entregas críticas.
## Eficiência frente às alternativas
Compare o algoritmo genético com os outros métodos usando os percentuais de \
`indicadores.comparativo`.
## Prioridades e cadeia fria
Como as entregas críticas e refrigeradas foram atendidas.
## Uso da frota
Ocupação e uso de autonomia por veículo; aponte veículos ociosos ou sobrecarregados.
## Pontos de atenção
Riscos operacionais observados nos dados."""

WEEKLY_REPORT = """Escreva o relatório SEMANAL de eficiência de rotas para a diretoria do \
hospital. Público: gestores — linguagem executiva.

<dados>{dados}</dados>

Seções obrigatórias:
## Resumo da semana
Três a cinco tópicos com os números mais relevantes (use `totais`).
## Economia gerada pela otimização
Quilômetros, reais e horas de condução economizados em relação ao método anterior \
(vizinho mais próximo), usando os valores já calculados.
## Evolução diária
Tabela Markdown com uma linha por dia (dia, entregas, km otimizado, km anterior, economia %, \
tempo médio até críticas).
## Qualidade do serviço
Entregas críticas, refrigeradas, violações de restrições.
## Recomendações
Duas ou três recomendações baseadas nos dados."""

SUGGESTIONS = """Analise o histórico operacional abaixo e sugira melhorias no processo de \
distribuição.

<dados>{dados}</dados>

Para cada sugestão (entre 3 e 5), use o formato:
### <título curto>
- **Evidência nos dados:** cite o número exato de `padroes` ou `totais` que motiva a sugestão.
- **Sugestão:** ação concreta.
- **Impacto esperado:** qualitativo; não invente números.
- **Como medir:** qual indicador acompanhar.

Ordene da maior para a menor relevância. Trate as sugestões como hipóteses a validar."""

# ------------------------------------------------------------------ chat

CHAT_SYSTEM = SYSTEM_BASE + """

Você responde perguntas em linguagem natural sobre as rotas e entregas do dia. Para obter \
fatos, chame SEMPRE as ferramentas disponíveis (não responda de memória). Respostas curtas e \
diretas, citando veículo, ordem da parada e horário previsto quando fizer sentido. Se a \
pergunta fugir de rotas, entregas, frota ou eficiência, diga educadamente que não é o seu escopo."""


CHAT_SYSTEM_CONTEXT = SYSTEM_BASE + """

Você responde perguntas em linguagem natural sobre as rotas e entregas do dia, usando exclusivamente os dados abaixo (índice de entregas críticas e refrigeradas, uma tabela por veículo com as paradas em ordem e o comparativo com outros métodos). Consulte o índice para perguntas sobre prioridade e as tabelas para os detalhes. Respostas curtas e diretas, citando veículo, ordem da parada e horário previsto quando fizer sentido. Se a pergunta fugir de rotas, entregas, frota ou eficiência, diga educadamente que não é o seu escopo.

<dados>{dados}</dados>"""


def render(template: str, data: dict) -> str:
    return template.format(dados=json.dumps(data, ensure_ascii=False, indent=1, default=str))
