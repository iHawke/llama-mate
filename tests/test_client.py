"""send_request() against a small mock OpenAI-compatible server (no real Ollama)."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llama_mate.client import format_summary, send_request

USAGE = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


class MockHandler(BaseHTTPRequestHandler):
    """Answers like an OpenAI-compatible server. The model name picks the behaviour."""

    last_headers = None

    def log_message(self, *args):  # keep test output quiet
        pass

    def send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        MockHandler.last_headers = dict(self.headers)
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))

        if payload["model"] == "missing":
            self.send_json(404, {"error": {"message": "model 'missing' not found"}})
        elif payload.get("n", 1) > 1:
            self.send_json(200, {
                "choices": [
                    {"index": i, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": f"Answer {i}"}}
                    for i in range(payload["n"])
                ],
                "usage": USAGE,
            })
        elif payload.get("stream"):
            self.send_stream()
        elif payload.get("tools"):
            self.send_json(200, {
                "choices": [{
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location": "Boston, MA"}'},
                        }],
                    },
                }],
                "usage": USAGE,
            })
        else:
            self.send_json(200, {
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello!", "reasoning": "Say hi."},
                }],
                "usage": USAGE,
            })

    def send_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": USAGE},
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture(scope="module")
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    server.shutdown()


def request(model="m", **extra):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}


def test_normal_reply(server_url):
    result = send_request(server_url, request())
    assert result.ok and result.status_code == 200
    choice = result.choices[0]
    assert choice["content"] == "Hello!"
    assert choice["reasoning"] == "Say hi."
    assert choice["finish_reason"] == "stop"
    assert result.usage == USAGE
    assert result.raw["choices"][0]["message"]["content"] == "Hello!"
    summary = format_summary(result)
    assert "Hello!" in summary and "total_tokens = 8" in summary and "HTTP 200" in summary


def test_several_choices(server_url):
    result = send_request(server_url, request(n=3))
    assert [choice["content"] for choice in result.choices] == ["Answer 0", "Answer 1", "Answer 2"]
    assert "Choice 2" in format_summary(result)


def test_tool_call(server_url):
    result = send_request(server_url, request(tools=[{"type": "function"}]))
    assert result.ok
    choice = result.choices[0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["tool_calls"][0]["function"]["name"] == "get_weather"
    summary = format_summary(result)
    assert "get_weather" in summary and "call_1" in summary


def test_sse_stream(server_url):
    result = send_request(server_url, request(stream=True))
    assert result.ok
    assert result.choices[0]["content"] == "Hello"
    assert result.choices[0]["finish_reason"] == "stop"
    assert result.usage == USAGE
    assert result.chunks[0].startswith("data: ")
    assert result.chunks[-1] == "data: [DONE]"
    assert "5 chunks" in format_summary(result)


def test_http_error(server_url):
    result = send_request(server_url, request(model="missing"))
    assert not result.ok
    assert result.status_code == 404
    assert "HTTP 404" in result.error and "not found" in result.error
    assert result.raw == {"error": {"message": "model 'missing' not found"}}


def test_unreachable_server():
    # Find a free port, then close it, so nothing is listening there.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = send_request(f"http://127.0.0.1:{port}/v1/chat/completions", request(), timeout=5)
    assert not result.ok
    assert "Could not reach the server" in result.error
    assert "Error" in format_summary(result)


def test_api_key_sent_as_bearer(server_url):
    send_request(server_url, request(), api_key="sk-test")
    assert MockHandler.last_headers["Authorization"] == "Bearer sk-test"


def test_no_api_key_no_header(server_url):
    send_request(server_url, request())
    assert "Authorization" not in MockHandler.last_headers
