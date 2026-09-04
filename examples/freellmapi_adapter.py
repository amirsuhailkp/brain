"""
Example concrete LLMInterface implementation for freellmapi
(github.com/tashfeenahmed/freellmapi) — a local OpenAI-compatible gateway
that fronts a pool of free-tier providers (Google AI Studio, Groq,
Cerebras, Mistral, OpenRouter, Cohere, Cloudflare Workers AI, Zhipu AI,
Ollama Cloud, HuggingFace Router, ...) behind one endpoint, with its own
health-checked fallback routing.

Same contract as every other adapter in this file (see AnthropicLLM /
OllamaLLM above): turn (prompt, schema_hint) -> dict. Brain's core never
knows or needs to know that one call actually lands on Groq and the next
on Mistral — that's freellmapi's job (its dashboard is what's showing
"1 healthy" per provider in the screenshot this adapter was built from),
not Brain's. Zero new dependency: plain stdlib urllib, same choice
OllamaLLM above already made, since freellmapi's chat endpoint is a
completely standard OpenAI-shaped POST.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from brain.interfaces import LLMInterface


class FreeLLMAPIError(Exception):
    pass


class FreeLLMAPILLM(LLMInterface):
    """
    `model` accepts either a literal freellmapi catalog id (e.g.
    "gemini-2.5-flash", "llama-3.3-70b-versatile" — whatever shows up on
    your dashboard's Keys/Models page) to pin one specific provider, or
    one of freellmapi's own routing strings to let IT choose across your
    enabled providers:

      "auto"           - follow whichever fallback chain is active on the
                          dashboard right now
      "auto:smart"     - favor the highest-intelligence enabled model
      "auto:fast"      - favor throughput / time-to-first-byte
      "auto:reliable"  - favor recent success rate
      "auto:<profile>" - route through a specific named dashboard profile

    This maps directly onto Brain's Phase 5 hybrid-model design with zero
    new code in core.py:

        hot    = FreeLLMAPILLM(model="auto:fast",   api_key=KEY)
        strong = FreeLLMAPILLM(model="auto:smart",  api_key=KEY)
        brain  = Brain(hot, project, brain_mem, project_mem, strong_llm=strong)

    Using two *routing policies* rather than two fixed model names is a
    better fit here than agent65's qwen3:4b/qwen3:8b split: freellmapi's
    own per-provider health tracking (the "1 healthy" badges) keeps
    adapting which actual provider serves each policy as they go up and
    down, so both Brain's hot path and its escalation path stay live even
    if any single one of the ten providers in the screenshot goes down.

    `api_key` defaults to reading `FREELLMAPI_KEY` from the environment
    (the unified key from the dashboard's Keys page) so it doesn't need
    to be hardcoded at the call site.
    """

    def __init__(
        self,
        model: str = "auto",
        base_url: str = "http://localhost:3001/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("FREELLMAPI_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def propose(self, prompt: str, schema_hint: str) -> dict:
        full_prompt = (
            f"{prompt}\n\nRespond with ONLY a JSON object matching this shape "
            f"and nothing else (no markdown fences, no prose): {schema_hint}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            # Pass-through hint (freellmapi docs/api.md): providers that
            # support strict JSON mode honor it; freellmapi's router
            # strips this for providers/models that don't (covered by
            # freellmapi's own test suite), so it's always safe to send
            # regardless of which provider "auto" happens to pick.
            "response_format": {"type": "json_object"},
        }
        body = self._post_with_retries(payload)
        text = self._extract_text(body)
        return self._parse_json(text)

    def _post_with_retries(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_err: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data, headers=headers
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")
                if e.code < 500:
                    # Bad model/profile name, bad key, malformed request -
                    # won't fix itself on retry. freellmapi's own error
                    # body is more useful here than a generic wrapper.
                    raise FreeLLMAPIError(f"freellmapi returned {e.code}: {detail}") from e
                last_err = FreeLLMAPIError(f"freellmapi returned {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = FreeLLMAPIError(f"freellmapi request failed: {e}")
        raise last_err  # exhausted retries on a 5xx or a network-level failure

    @staticmethod
    def _extract_text(body: dict) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise FreeLLMAPIError(f"freellmapi response missing expected shape: {body}") from e

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # LLMInterface contract (see AnthropicLLM/OllamaLLM above):
            # raise on unrecoverable failure, don't silently guess -
            # PlanningEngine.reason_and_plan() already catches this and
            # degrades to an empty proposal for the step.
            raise ValueError(
                f"FreeLLMAPILLM: model did not return valid JSON: {e}\nRaw: {text[:500]}"
            ) from e
