# Relatório Diário de Eficiência das Rotas
**Destinatários:** Coordenação de Logística e Diretoria do Hospital  
**Turno de Operação:** 08:00 - 17:00  
**Método de Otimização:** AG — E9_memetico (melhor de 5 seeds)

---

## Resumo executivo
* **Viabilidade e Conformidade Integral:** A operação do dia foi concluída com 100% de viabilidade técnica. Não foram registradas violações de capacidade de carga, limite de autonomia dos veículos, janelas de tempo de entrega ou restrições de refrigeração.
* **Eficiência de Custos e Distância:** Foram realizadas 40 entregas com uma distância total percorrida de 264.08 km, gerando um custo operacional de R$ 424.39 (custo objetivo de 449.62).
* **Garantia de Fluxo de Urgência:** O tempo médio para o atendimento das entregas de prioridade crítica foi de 2.17 horas, assegurando agilidade no abastecimento dos setores mais sensíveis.

---

## Indicadores do dia

| Indicador | Valor Registrado |
| :--- | :--- |
| **Distância Total Percorrida** | 264.08 km |
| **Custo Operacional** | R$ 424.39 |
| **Veículos Utilizados** | 4 |
| **Viabilidade da Rota** | Sim (Viável) |
| **Tempo Médio até Entregas Críticas** | 2.17 h |
| **Penalidades Aplicadas** | 0.0 |

---

## Eficiência frente às alternativas

O algoritmo genético utilizado (**AG — E9_memetico**) apresentou desempenho superior à maioria dos métodos alternativos testados:

* **OR-Tools (GLS 30s):** O AG obteve uma economia de distância de 0.2%, embora tenha apresentado um custo objetivo ligeiramente superior em -0.1%. Ambos os métodos geraram rotas viáveis.
* **AG — E1_base (melhor seed):** O AG atual superou a versão base, gerando uma economia de 3.5% em distância e 3.1% no custo objetivo. Ambos viáveis.
* **Clarke-Wright (savings):** O AG gerou uma economia de 15.9% em distância e 54.1% no custo objetivo. O método Clarke-Wright gerou uma rota **inviável**.
* **Vizinho mais próximo:** O AG proporcionou uma economia de 28.9% em distância e 72.9% no custo objetivo. O método do Vizinho mais próximo gerou uma rota **inviável**.
* **Aleatório (melhor de 2000):** O AG obteve economia de 58.1% em distância e 97.2% no custo objetivo. O método aleatório gerou uma rota **inviável**.
* **Código base (TSP, 1 veículo):** O AG gerou economia de 15.1% em distância e 99.5% no custo objetivo. O código base gerou uma rota **inviável**.

---

## Prioridades e cadeia fria

O transporte de itens de **prioridade crítica** e que exigem **cadeia fria (refrigerados)** foi priorizado e executado sem nenhuma violação de temperatura (0 ocorrências de falha de refrigeração):

* **Cadeia Fria (Itens Refrigerados):** Atendida exclusivamente pelas vans refrigeradas, totalizando 13 entregas sob controle térmico.
  * **Van Refrigerada 1:** Transportou 6 itens refrigerados.
  * **Van Refrigerada 2:** Transportou 7 itens refrigerados.
* **Entregas Críticas (Urgências/UTI):** Foram realizadas por múltiplos veículos para otimização do tempo de resposta.
  * **Van Refrigerada 1:** 4 entregas críticas.
  * **Van Refrigerada 2:** 3 entregas críticas.
  * **Van Carga 3:** 1 entrega crítica.
  * **Moto 5:** 1 entrega crítica.
  * **Tempo máximo de espera para entrega crítica:** 5.68 horas.

---

## Uso da frota

* **Van Refrigerada 1 (Em operação):**
  * Carga total: 311.3 kg
  * Ocupação: 77.8%
  * Distância percorrida: 65.43 km
  * Uso de autonomia: 36.4%
  * Retorno previsto: 13:31
  * Custo: R$ 117.77 (12 paradas)
* **Van Refrigerada 2 (Em operação):**
  * Carga total: 326.9 kg
  * Ocupação: 81.7%
  * Distância percorrida: 118.66 km
  * Uso de autonomia: 65.9%
  * Retorno previsto: 16:12
  * Custo: R$ 213.59 (15 paradas)
* **Van Carga 3 (Em operação):**
  * Carga total: 204.6 kg
  * Ocupação: 34.1%
  * Distância percorrida: 53.03 km
  * Uso de autonomia: 26.5%
  * Retorno previsto: 12:36
  * Custo: R$ 79.54 (11 paradas)
* **Moto 5 (Em operação):**
  * Carga total: 7.1 kg
  * Ocupação: 28.4%
  * Distância percorrida: 26.96 km
  * Uso de autonomia: 22.5%
  * Retorno previsto: 11:36
  * Custo: R$ 13.48 (2 paradas)
* **Moto 4 (Ociosa):**
  * Não utilizada na operação do dia (0.0% de ocupação, 0.0 km percorridos, custo R$ 0.0).

---

## Pontos de atenção

1. **Janela de Retorno Limítrofe:** A **Van Refrigerada 2** possui retorno previsto para as 16:12, o que se aproxima do término do turno operacional (17:00). Qualquer imprevisto no trânsito ou no ato da entrega pode gerar horas extras.
2. **Concentração de Demanda na Van Refrigerada 2:** Este veículo operou com a maior taxa de ocupação de carga (81.7%), maior distância percorrida (118.66 km) e maior uso de autonomia (65.9%) do dia, concentrando 15 paradas.
3. **Ociosidade de Recurso:** O veículo **Moto 4** permaneceu totalmente ocioso durante o turno, indicando que a demanda do dia foi absorvida pelos outros 4 veículos ativos sem necessidade de acionamento deste recurso.
4. **Tempo Máximo de Espera Crítica:** Embora o tempo médio de atendimento crítico seja satisfatório (2.17 horas), o tempo máximo registrado de 5.68 horas exige monitoramento para garantir que itens de alta urgência não fiquem retidos por períodos prolongados em rotas longas.