"""
Offline verification for examples/freellmapi_adapter.py against a mock
server that mimics freellmapi's OpenAI-compatible /v1/chat/completions
shape (success, code-fenced JSON, 4xx fail-fast, 5xx-then-retry,
malformed JSON). Loopback only - no external network required, so this
runs fine in any sandboxed environment, unlike a real freellmapi call.

Not part of the main pytest suite (examples/ is explicitly dev-only, same
as claude_llm_adapter.py's docstring already states "not wired into
tests"); run directly: `python3 tests/verify_freellmapi_adapter.py`
"""
import http.server
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from freellmapi_adapter import FreeLLMAPIError, FreeLLMAPILLM

PASS = []
FAIL = []


def check(name, condition):
    (PASS if condition else FAIL).append(name)
    print(("  OK  " if condition else " FAIL ") + name)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep test output clean

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        route = self.server.route_table.get(body["model"], "plain")

        if route == "fenced":
            content = '```json\n{"hypotheses": [], "candidate_actions": []}\n```'
            self._send(200, {"choices": [{"message": {"content": content}}]})
        elif route == "malformed":
            self._send(200, {"choices": [{"message": {"content": "not json at all"}}]})
        elif route == "bad_model":
            self._send(400, {"error": "unknown model or profile"})
        elif route == "flaky_500":
            self.server.flaky_calls = getattr(self.server, "flaky_calls", 0) + 1
            if self.server.flaky_calls < 2:
                self._send(500, {"error": "upstream provider timeout"})
            else:
                content = '{"hypotheses": [], "candidate_actions": []}'
                self._send(200, {"choices": [{"message": {"content": content}}]})
        else:
            content = '{"hypotheses": [], "candidate_actions": []}'
            self._send(200, {"choices": [{"message": {"content": content}}]})

    def _send(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


def main():
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    server.route_table = {
        "plain-model": "plain",
        "fenced-model": "fenced",
        "malformed-model": "malformed",
        "bad-model": "bad_model",
        "flaky-model": "flaky_500",
    }
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}/v1"

    # 1. plain JSON response parses correctly
    llm = FreeLLMAPILLM(model="plain-model", base_url=base_url, api_key="test-key")
    result = llm.propose("do the thing", '{"hypotheses": [], "candidate_actions": []}')
    check("plain JSON response parses to dict", result == {"hypotheses": [], "candidate_actions": []})

    # 2. code-fenced JSON (some providers wrap in ```json even when asked not to) still parses
    llm = FreeLLMAPILLM(model="fenced-model", base_url=base_url, api_key="test-key")
    result = llm.propose("do the thing", "{}")
    check("code-fenced JSON still parses", result == {"hypotheses": [], "candidate_actions": []})

    # 3. malformed content raises ValueError (LLMInterface contract), not a silent guess
    llm = FreeLLMAPILLM(model="malformed-model", base_url=base_url, api_key="test-key")
    try:
        llm.propose("do the thing", "{}")
        check("malformed JSON raises ValueError", False)
    except ValueError:
        check("malformed JSON raises ValueError", True)

    # 4. a 4xx (bad model/profile name) fails fast, no retry loop
    llm = FreeLLMAPILLM(model="bad-model", base_url=base_url, api_key="test-key", max_retries=3)
    try:
        llm.propose("do the thing", "{}")
        check("4xx raises FreeLLMAPIError", False)
    except FreeLLMAPIError as e:
        check("4xx raises FreeLLMAPIError", "400" in str(e))

    # 5. a transient 5xx is retried and eventually succeeds
    llm = FreeLLMAPILLM(model="flaky-model", base_url=base_url, api_key="test-key", max_retries=2)
    result = llm.propose("do the thing", "{}")
    check("transient 5xx recovers via retry", result == {"hypotheses": [], "candidate_actions": []})

    # 6. api_key falls back to env var when not passed explicitly
    import os

    os.environ["FREELLMAPI_KEY"] = "from-env"
    llm = FreeLLMAPILLM(model="plain-model", base_url=base_url)
    check("api_key defaults from FREELLMAPI_KEY env var", llm.api_key == "from-env")
    del os.environ["FREELLMAPI_KEY"]

    server.shutdown()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
