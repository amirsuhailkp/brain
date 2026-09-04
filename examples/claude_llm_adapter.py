"""
Example concrete LLMInterface implementation. Not wired into tests (no
network access there) — this is the reference for how agent65's local
Ollama client, or a hosted Claude/GPT client, plugs into the Brain.

Anything that can turn (prompt, schema_hint) -> dict satisfies the contract.
Swapping this for Ollama means: same Brain code, different adapter file.
"""
from __future__ import annotations

import json
import re

from brain.interfaces import LLMInterface


class AnthropicLLM(LLMInterface):
    def __init__(self, client, model: str = "claude-sonnet-4-6"):
        """`client` is any object with a `.messages.create(...)`-shaped
        method (e.g. the `anthropic` SDK's Anthropic() client)."""
        self.client = client
        self.model = model

    def propose(self, prompt: str, schema_hint: str) -> dict:
        full_prompt = (
            f"{prompt}\n\nRespond with ONLY a JSON object matching this shape "
            f"and nothing else (no markdown fences, no prose): {schema_hint}"
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        return json.loads(cleaned)


class OllamaLLM(LLMInterface):
    """Reference adapter for Amir's local Ollama setup — same contract,
    zero changes needed to brain/core.py."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host = host
        self.model = model

    def propose(self, prompt: str, schema_hint: str) -> dict:
        import urllib.request

        full_prompt = (
            f"{prompt}\n\nRespond with ONLY a JSON object matching this shape "
            f"and nothing else: {schema_hint}"
        )
        payload = json.dumps(
            {"model": self.model, "prompt": full_prompt, "stream": False, "format": "json"}
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        return json.loads(body["response"])
