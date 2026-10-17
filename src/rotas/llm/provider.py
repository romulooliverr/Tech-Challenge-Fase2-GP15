"""Provedores de LLM.

* ``GeminiProvider`` — Google Gemini via SDK ``google-genai`` (plano gratuito do AI Studio),
  com cadeia de modelos de reserva e espera exponencial para limites por minuto.
* ``OllamaProvider`` — modelo local via Ollama (``ollama serve``), sem cota nem internet.
* ``MockProvider``   — offline e determinístico; usado em testes e quando não há chave.

Seleção por variável de ambiente: ``LLM_PROVIDER=gemini|ollama|mock`` (padrão: gemini se houver
``GEMINI_API_KEY``, senão mock), ``GEMINI_MODEL`` (padrão ``gemini-3.5-flash``),
``OLLAMA_MODEL`` (padrão ``qwen2.5:14b``) e ``OLLAMA_BASE_URL`` (padrão ``http://localhost:11434``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from typing import Protocol

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class ChatSession(Protocol):
    def send(self, message: str) -> str: ...


class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, prompt: str, *, json_output: bool = False, temperature: float = 0.3) -> str: ...

    def start_chat(self, system: str, tools: list[Callable]) -> ChatSession: ...


# ------------------------------------------------------------------------------ Gemini

DEFAULT_FALLBACKS = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"


class QuotaExhausted(RuntimeError):
    """Cota diária do plano gratuito esgotada em todos os modelos configurados."""


def _is_daily_quota(e) -> bool:
    return e.code == 429 and "PerDay" in str(e)


class GeminiProvider:
    """Gemini com cadeia de modelos: se a cota DIÁRIA de um modelo acaba (o plano gratuito
    do gemini-3.5-flash permite poucas requisições/dia), passa para o próximo da lista
    (``GEMINI_FALLBACK_MODELS``). Limites por minuto (429) e instabilidade (5xx) são
    tratados com espera exponencial."""

    name = "gemini"
    _exhausted: set[str] = set()  # compartilhado no processo

    def __init__(self, model: str | None = None, api_key: str | None = None, fallbacks: list[str] | None = None) -> None:
        from google import genai

        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        if fallbacks is None:
            fallbacks = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",") if m.strip()]
        self.models = list(dict.fromkeys([self.model, *fallbacks]))
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY não definida. Gere uma chave gratuita em "
                "https://aistudio.google.com/apikey e coloque no arquivo .env"
            )
        self.client = genai.Client(api_key=key)
        self.last_model: str | None = None

    def _call(self, fn: Callable[[str], object], retries: int = 4, base_delay: float = 8.0):
        """Executa ``fn(modelo)`` percorrendo a cadeia de modelos."""
        import httpx
        from google.genai import errors

        for model in [m for m in self.models if m not in self._exhausted]:
            for attempt in range(retries + 1):
                try:
                    out = fn(model)
                    self.last_model = model
                    return out
                except httpx.TransportError as e:  # queda de conexão, timeout de rede
                    if attempt == retries:
                        raise
                    logger.warning("Erro de rede (%s) — nova tentativa em %.0fs", type(e).__name__, base_delay)
                    time.sleep(base_delay)
                except errors.APIError as e:
                    if _is_daily_quota(e):
                        logger.warning("Cota diária esgotada para %s — tentando o próximo modelo", model)
                        self._exhausted.add(model)
                        break
                    retryable = e.code == 429 or (e.code is not None and e.code >= 500)
                    if not retryable:
                        raise
                    if attempt == retries:
                        logger.warning("%s indisponível após %d tentativas — próximo modelo", model, retries)
                        break
                    delay = base_delay * 2**attempt
                    logger.warning("Gemini %s em %s (tentativa %d/%d) — aguardando %.0fs", e.code, model, attempt + 1, retries, delay)
                    time.sleep(delay)
        raise QuotaExhausted(
            f"Sem cota disponível nos modelos {self.models}. O plano gratuito renova diariamente; "
            "ajuste GEMINI_MODEL/GEMINI_FALLBACK_MODELS ou use LLM_PROVIDER=mock."
        )

    def generate(self, system: str, prompt: str, *, json_output: bool = False, temperature: float = 0.3) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            response_mime_type="application/json" if json_output else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self._call(lambda m: self.client.models.generate_content(model=m, contents=prompt, config=config))
        return resp.text or ""

    def start_chat(self, system: str, tools: list[Callable]) -> ChatSession:
        """Chat com chamada automática de ferramentas (funções Python).

        Cada chamada de ferramenta consome uma requisição extra da cota."""
        from google.genai import types

        provider = self
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            tools=tools or None,
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(maximum_remote_calls=6)
                if tools
                else types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )

        class _Session:
            def __init__(self) -> None:
                self.history: list = []

            def send(self, message: str) -> str:
                def run(model: str):
                    # recria o chat com o histórico — permite trocar de modelo no meio da conversa
                    chat = provider.client.chats.create(model=model, config=config, history=self.history)
                    resp = chat.send_message(message)
                    self.history = chat.get_history()
                    return resp

                return provider._call(run).text or ""

        return _Session()

# ------------------------------------------------------------------------------ Ollama

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
_PY_TYPES = {"int": int, "float": float, "str": str}
MAX_TOOL_ROUNDS = 10


def _annotation_name(annotation) -> str:
    # com ``from __future__ import annotations`` a anotação chega como texto ("int")
    return annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))


def _tool_schema(tool: Callable) -> dict:
    """Schema JSON (formato OpenAI/Ollama) a partir da assinatura e docstring da função."""
    import inspect

    properties, required = {}, []
    for name, param in inspect.signature(tool).parameters.items():
        properties[name] = {"type": _JSON_TYPES.get(_annotation_name(param.annotation), "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": inspect.getdoc(tool) or f"Executa a ferramenta {tool.__name__}.",
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _call_tool(tool: Callable, arguments: dict) -> str:
    """Executa a ferramenta convertendo os argumentos para o tipo anotado (modelos locais às
    vezes mandam ``"3"`` em vez de ``3``). Erros voltam como texto para o modelo."""
    import inspect

    try:
        params = inspect.signature(tool).parameters
        kwargs = {}
        for k, v in (arguments or {}).items():
            cast = _PY_TYPES.get(_annotation_name(params[k].annotation)) if k in params else None
            kwargs[k] = cast(v) if cast and not isinstance(v, cast) else v
        result = tool(**kwargs)
    except (TypeError, ValueError, KeyError) as e:
        return f"Erro ao executar {tool.__name__}: {e}"
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


class OllamaProvider:
    """Modelos locais servidos pelo Ollama (``ollama serve``). Sem cota e sem internet;
    a qualidade depende do modelo baixado (``ollama pull qwen2.5:14b``)."""

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        import ollama

        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.host = host or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        self.client = ollama.Client(host=self.host)

    def _chat(self, **kwargs):
        import ollama

        try:
            return self.client.chat(model=self.model, stream=False, **kwargs)
        except ConnectionError as e:
            raise RuntimeError(
                f"Ollama não está respondendo em {self.host}. Rode `ollama serve` (e "
                f"`ollama pull {self.model}`) ou use LLM_PROVIDER=gemini|mock."
            ) from e
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise RuntimeError(f"Modelo {self.model!r} não encontrado. Rode `ollama pull {self.model}`.") from e
            raise

    def generate(self, system: str, prompt: str, *, json_output: bool = False, temperature: float = 0.3) -> str:
        resp = self._chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            options={"temperature": temperature},
            format="json" if json_output else None,
        )
        return resp.message.content or ""

    def start_chat(self, system: str, tools: list[Callable]) -> ChatSession:
        """Chat com chamada de ferramentas (funções Python), em laço até o modelo responder em texto."""
        provider = self
        by_name = {t.__name__: t for t in tools}
        schemas = [_tool_schema(t) for t in tools] or None

        class _Session:
            def __init__(self) -> None:
                self.messages: list[dict] = [{"role": "system", "content": system}]

            def send(self, message: str) -> str:
                self.messages.append({"role": "user", "content": message})
                for _ in range(MAX_TOOL_ROUNDS):  # evita laço infinito de ferramentas
                    msg = provider._chat(messages=self.messages, tools=schemas, options={"temperature": 0.2}).message
                    calls = msg.tool_calls or []
                    self.messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        **({"tool_calls": [{"function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]} if calls else {}),
                    })
                    if not calls:
                        return msg.content or ""
                    for c in calls:
                        tool = by_name.get(c.function.name)
                        result = _call_tool(tool, c.function.arguments) if tool else f"Erro: ferramenta {c.function.name!r} não existe."
                        self.messages.append({"role": "tool", "tool_name": c.function.name, "content": result})
                return f"Não consegui concluir a resposta após {MAX_TOOL_ROUNDS} rodadas de chamadas de ferramentas."

        return _Session()


# ------------------------------------------------------------------------------ Mock


class MockProvider:
    """Respostas determinísticas montadas a partir do próprio contexto (sem rede).

    Não "entende" linguagem: serve para testar o encanamento (prompts, ferramentas, avaliação)
    e para demonstrar o sistema offline.
    """

    name = "mock"

    def generate(self, system: str, prompt: str, *, json_output: bool = False, temperature: float = 0.3) -> str:
        ctx = _extract_json(prompt)
        if json_output:
            return json.dumps({"resposta": "mock", "chaves_contexto": sorted(ctx)[:10]}, ensure_ascii=False)
        lines = ["[MOCK] Resposta gerada offline a partir dos dados fornecidos.", ""]
        if "paradas" in ctx:  # instruções de motorista: lista as paradas na ordem
            lines.append(f"## Rota — {ctx.get('veiculo')}")
            for s in ctx["paradas"]:
                flags = (" [CRÍTICO]" if s["prioridade"] == "crítica" else "") + (" [REFRIGERADO]" if s["refrigerado"] else "")
                lines.append(f"{s['ordem']}. {s['destino']} — chegada {s['chegada_prevista']}{flags}")
        elif "resumo" in ctx:
            for k, v in ctx["resumo"].items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append(json.dumps(ctx, ensure_ascii=False)[:800])
        return "\n".join(lines)

    def start_chat(self, system: str, tools: list[Callable]) -> ChatSession:
        by_name = {t.__name__: t for t in tools}

        class _Session:
            def send(self, message: str) -> str:
                # roteamento ingênuo por palavra-chave, só para exercitar as ferramentas
                msg = message.lower()
                if "metricas_gerais" in by_name and any(w in msg for w in ("total", "resumo", "distância", "custo")):
                    return "[MOCK] " + json.dumps(by_name["metricas_gerais"](), ensure_ascii=False)
                if "buscar_entrega" in by_name:
                    for word in message.replace("?", " ").split():
                        if len(word) > 3:
                            found = by_name["buscar_entrega"](word)
                            if found.get("resultados"):
                                return "[MOCK] " + json.dumps(found, ensure_ascii=False)[:1500]
                return "[MOCK] Não consegui interpretar a pergunta sem um LLM real."

        return _Session()


def _extract_json(text: str) -> dict:
    """Primeiro objeto JSON dentro de <dados>...</dados> (convenção dos prompts)."""
    start, end = text.find("<dados>"), text.find("</dados>")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start + len("<dados>") : end])
    except json.JSONDecodeError:
        return {}


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or os.getenv("LLM_PROVIDER") or "").lower()
    if not name:
        name = "gemini" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "mock"
    if name == "gemini":
        return GeminiProvider()
    if name == "ollama":
        return OllamaProvider()
    if name == "mock":
        return MockProvider()
    raise ValueError(f"LLM_PROVIDER desconhecido: {name!r} (use 'gemini', 'ollama' ou 'mock')")
