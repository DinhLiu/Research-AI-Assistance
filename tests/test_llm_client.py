import httpx
import pytest

from research_assistant.llm import client
from research_assistant.llm.governor import Quota, RunBudget, LlmError
from tests.test_llm_governor import setup


@pytest.mark.parametrize("provider", ["gemini", "openai", "groq"])
def test_adapter_output_cap_usage_and_single_attempt(tmp_path, monkeypatch, provider):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "test-not-a-key")
    calls = []
    def post(self, url, **kwargs):
        calls.append(kwargs)
        if provider == "gemini":
            assert kwargs["json"]["generationConfig"]["maxOutputTokens"] == 123
            data = {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 2}}
        else:
            assert kwargs["json"]["max_tokens"] == 123
            data = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2}}
        return httpx.Response(200, json=data)
    monkeypatch.setattr(httpx.Client, "post", post)
    _, gov = setup(tmp_path)
    result = client.complete_result(system="test", user="test", provider=provider,
                                    max_output_tokens=123, governor=gov,
                                    quota=Quota(60, 10000, 1000), budget=RunBudget("run"))
    assert len(calls) == 1
    assert result.input_tokens == 12
    assert result.output_tokens == 2
    assert result.text == "{}"


def test_client_retry_is_paced_and_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-key")
    clock, gov = setup(tmp_path)
    starts = []
    def post(self, *args, **kwargs):
        starts.append(clock.now)
        if len(starts) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})
    monkeypatch.setattr(httpx.Client, "post", post)
    assert client.complete(system="test", user="test", provider="gemini", governor=gov,
                           quota=Quota(60, 20000, 1000), budget=RunBudget("run")) == "{}"
    assert starts == [1000, 1015]
    assert gov.stats("run")["http_attempts"] == 2
