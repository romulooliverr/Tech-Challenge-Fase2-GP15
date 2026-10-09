"""Avaliação automática da qualidade dos textos gerados pela LLM.

Métricas (todas determinísticas, sem LLM):

* **fidelidade numérica** — fração dos números citados no texto que existem nos dados
  (detecta alucinação de horários, pesos e distâncias);
* **cobertura** — fração das paradas citadas pelo nome;
* **ordem** — as paradas aparecem na mesma ordem da rota?
* **destaques** — toda parada crítica/refrigerada vem acompanhada do marcador adequado?
* **estrutura** — as seções exigidas pelo prompt estão presentes?

Opcionalmente, um **LLM-juiz** atribui notas de 1 a 5 (clareza, completude, utilidade).
"""

from __future__ import annotations

import json
import re
import unicodedata

from rotas.llm.provider import LLMProvider

NUM_RE = re.compile(r"(?<![\w#])(\d{1,2}:\d{2}|\d+(?:[.,]\d+)?)(?![\w])")
DRIVER_SECTIONS = ["Resumo da rota", "Antes de sair", "Paradas", "Atenção", "Em caso de imprevisto"]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _collect_numbers(obj, out: set[str]) -> set[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, int | float):
        out.add(_canon(str(obj)))
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            out.add(_canon(m))
    return out


def _canon(tok: str) -> str:
    if ":" in tok:
        h, m = tok.split(":")
        return f"{int(h):02d}:{m}"
    try:
        x = float(tok.replace(",", "."))
    except ValueError:
        return tok
    return f"{x:.2f}".rstrip("0").rstrip(".")


def numeric_faithfulness(text: str, data: dict, ignore_small_ints: int = 12) -> dict:
    """Números do texto que não aparecem nos dados são possíveis alucinações.

    Inteiros pequenos (ordens de parada, contagens) e números das regras do prompt
    (2 °C a 8 °C) são ignorados.
    """
    known = _collect_numbers(data, set()) | {"2", "8"}
    cited = [_canon(t) for t in NUM_RE.findall(text)]
    checked = [
        c for c in cited
        if ":" in c or not (c.replace(".", "").isdigit() and "." not in c and float(c) <= ignore_small_ints)
    ]
    unsupported = sorted({c for c in checked if c not in known})
    score = 1 - len([c for c in checked if c not in known]) / len(checked) if checked else 1.0
    return {"score": round(score, 3), "numeros_verificados": len(checked), "nao_suportados": unsupported}


def stop_coverage_and_order(text: str, route: dict) -> dict:
    t = _norm(text)
    positions = []
    missing = []
    for s in route["paradas"]:
        pos = t.find(_norm(s["destino"]))
        if pos == -1:
            missing.append(s["destino"])
        else:
            # considera a última menção "em lista de paradas": procura a partir da seção Paradas
            sec = t.find("paradas")
            later = t.find(_norm(s["destino"]), sec if sec != -1 else 0)
            positions.append(later if later != -1 else pos)
    in_order = all(a < b for a, b in zip(positions, positions[1:]))
    n = len(route["paradas"])
    return {"cobertura": round((n - len(missing)) / n, 3) if n else 1.0, "ordem_correta": in_order, "faltando": missing}


def highlight_check(text: str, route: dict, window: int = 350) -> dict:
    t = _norm(text)
    sec = t.find("## paradas")
    errors = []
    checks = 0
    for s in route["paradas"]:
        pos = t.find(_norm(s["destino"]), sec if sec != -1 else 0)
        if pos == -1:
            continue
        snippet = t[pos : pos + window]
        if s["prioridade"] == "crítica":
            checks += 1
            if "critic" not in snippet:
                errors.append(f"{s['destino']}: sem destaque de crítica")
        if s["refrigerado"]:
            checks += 1
            if "refriger" not in snippet and "❄" not in snippet:
                errors.append(f"{s['destino']}: sem destaque de refrigerado")
    return {"score": round(1 - len(errors) / checks, 3) if checks else 1.0, "erros": errors}


def section_check(text: str, sections: list[str]) -> dict:
    t = _norm(text)
    missing = [s for s in sections if _norm(s) not in t]
    return {"score": round(1 - len(missing) / len(sections), 3), "faltando": missing}


def evaluate_driver_text(text: str, route: dict) -> dict:
    num = numeric_faithfulness(text, route)
    cov = stop_coverage_and_order(text, route)
    hl = highlight_check(text, route)
    sec = section_check(text, DRIVER_SECTIONS)
    overall = (num["score"] + cov["cobertura"] + float(cov["ordem_correta"]) + hl["score"] + sec["score"]) / 5
    return {"geral": round(overall, 3), "fidelidade_numerica": num, "cobertura_ordem": cov, "destaques": hl, "estrutura": sec}


JUDGE_PROMPT = """Você é um avaliador rigoroso de textos operacionais de logística hospitalar.
Avalie as INSTRUÇÕES PARA MOTORISTA abaixo em relação aos DADOS da rota.

<dados>{dados}</dados>

<texto>{texto}</texto>

Responda APENAS um JSON com notas inteiras de 1 a 5 e uma justificativa curta:
{{"clareza": n, "completude": n, "correcao": n, "utilidade": n, "justificativa": "..."}}"""


def llm_judge(provider: LLMProvider, text: str, route: dict) -> dict:
    raw = provider.generate(
        "Você avalia textos de forma objetiva.",
        JUDGE_PROMPT.format(dados=json.dumps(route, ensure_ascii=False), texto=text),
        json_output=True,
        temperature=0.0,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {"erro": raw[:200]}
