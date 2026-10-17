"""OllamaProvider com um client falso: sem servidor Ollama nem rede."""

from types import SimpleNamespace

import pytest

from rotas.llm.provider import OllamaProvider, get_provider


def _resp(content="", calls=()):
    tool_calls = [SimpleNamespace(function=SimpleNamespace(name=n, arguments=a)) for n, a in calls] or None
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))


class FakeClient:
    def __init__(self, answers):
        self.answers, self.requests = list(answers), []

    def chat(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # cópia: a sessão segue alterando a lista
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _provider(answers) -> OllamaProvider:
    p = OllamaProvider(model="teste", host="http://ollama.test")
    p.client = FakeClient(answers)
    return p


def test_get_provider_ollama():
    assert get_provider("ollama").name == "ollama"


def test_generate_sends_system_prompt_and_json_format():
    p = _provider([_resp('{"ok": 1}')])
    assert p.generate("sis", "pergunta", json_output=True) == '{"ok": 1}'
    req = p.client.requests[0]
    assert req["model"] == "teste" and req["format"] == "json"
    assert [m["role"] for m in req["messages"]] == ["system", "user"]


def test_chat_runs_tool_and_coerces_argument_type():
    seen = []

    def rota_do_veiculo(veiculo_id: int) -> dict:
        """Rota de um veículo."""
        seen.append(veiculo_id)
        return {"veiculo_id": veiculo_id}

    p = _provider([_resp(calls=[("rota_do_veiculo", {"veiculo_id": "2"})]), _resp("Veículo 2 ok")])
    chat = p.start_chat("sis", [rota_do_veiculo])
    assert chat.send("rota do 2?") == "Veículo 2 ok"
    assert seen == [2]  # "2" (texto) virou int
    schema = p.client.requests[0]["tools"][0]["function"]
    assert schema["name"] == "rota_do_veiculo" and schema["parameters"]["properties"]["veiculo_id"]["type"] == "integer"
    tool_msg = p.client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and '"veiculo_id": 2' in tool_msg["content"]


def test_unknown_tool_is_reported_to_the_model():
    p = _provider([_resp(calls=[("nao_existe", {})]), _resp("desculpe")])
    assert p.start_chat("sis", []).send("oi") == "desculpe"
    assert "não existe" in p.client.requests[1]["messages"][-1]["content"]


def test_server_down_gives_clear_error():
    p = _provider([ConnectionError("recusado")])
    with pytest.raises(RuntimeError, match="ollama serve"):
        p.generate("sis", "oi")
