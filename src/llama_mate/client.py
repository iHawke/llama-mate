"""Send a request to an OpenAI-compatible server and collect the response.

There is no Gradio in this file. It handles:
  * normal replies (one JSON object),
  * streaming replies (Server-Sent Events, "data: {...}" lines),
  * errors: server unreachable, timeouts, HTTP errors and bad JSON.
"""

import json
from dataclasses import dataclass, field

import requests

from . import defaults


@dataclass
class Result:
    """Everything we learned from one request."""

    ok: bool = True
    error: str = ""
    status_code: int | None = None
    choices: list = field(default_factory=list)  # one entry per answer (see new_choice)
    usage: dict | None = None
    raw: dict | None = None  # the full JSON reply (non-streaming only)
    chunks: list = field(default_factory=list)  # raw SSE lines (streaming only)


def new_choice(index):
    """One answer. You get several when the request sends n > 1."""
    return {
        "index": index,
        "content": "",
        "reasoning": "",
        "refusal": None,
        "tool_calls": [],
        "finish_reason": None,
    }


def get_choice(result, index):
    """Find the choice with this index, creating it (and any before it) if needed."""
    while len(result.choices) <= index:
        result.choices.append(new_choice(len(result.choices)))
    return result.choices[index]


def make_headers(api_key=None):
    """JSON headers, plus the API key for cloud servers. The key is never printed."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def error_message(response):
    """Pull a readable message out of an HTTP error reply."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or response.reason
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)
    return str(error)


def read_reply(result, data):
    """Fill `result` from a normal (non-streaming) reply."""
    result.raw = data
    result.usage = data.get("usage")
    for position, item in enumerate(data.get("choices") or []):
        choice = get_choice(result, item.get("index", position))
        message = item.get("message") or {}
        choice["content"] = message.get("content") or ""
        # Servers name the "thinking" text differently (Ollama: reasoning).
        choice["reasoning"] = message.get("reasoning") or message.get("reasoning_content") or ""
        choice["refusal"] = message.get("refusal")
        choice["tool_calls"] = message.get("tool_calls") or []
        choice["finish_reason"] = item.get("finish_reason")


def read_chunk(result, chunk):
    """Add one streamed chunk to `result`. Each chunk holds a small "delta"."""
    if chunk.get("usage"):
        result.usage = chunk["usage"]  # only sent when stream_options.include_usage is true
    for item in chunk.get("choices") or []:
        choice = get_choice(result, item.get("index", 0))
        delta = item.get("delta") or {}
        choice["content"] += delta.get("content") or ""
        choice["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""

        # Tool calls arrive in pieces, matched by their "index".
        for piece in delta.get("tool_calls") or []:
            calls = choice["tool_calls"]
            number = piece.get("index", len(calls))
            while len(calls) <= number:
                calls.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            call = calls[number]
            if piece.get("id"):
                call["id"] = piece["id"]
            function = piece.get("function") or {}
            call["function"]["name"] += function.get("name") or ""
            call["function"]["arguments"] += function.get("arguments") or ""

        if item.get("finish_reason"):
            choice["finish_reason"] = item["finish_reason"]


def send_request_live(url, payload, api_key=None, timeout=defaults.REQUEST_TIMEOUT):
    """Send the request and yield the Result as it grows.

    For a normal reply this yields once. For a stream it yields after every chunk,
    so the UI can show the answer while it is being written.
    """
    streaming = bool(payload.get("stream"))
    result = Result()

    try:
        response = requests.post(
            url, json=payload, headers=make_headers(api_key), stream=streaming, timeout=timeout
        )
    except requests.ConnectionError:
        result.ok = False
        result.error = (
            f"Could not reach the server at {url}. Is it running? "
            "For Ollama, start it with `ollama serve` (or open the Ollama app)."
        )
        yield result
        return
    except requests.Timeout:
        result.ok = False
        result.error = f"The server did not answer within {timeout} seconds."
        yield result
        return
    except requests.RequestException as error:  # e.g. a badly written URL
        result.ok = False
        result.error = f"Could not send the request: {error}"
        yield result
        return

    result.status_code = response.status_code
    if response.status_code >= 400:
        result.ok = False
        result.error = f"HTTP {response.status_code} from the server: {error_message(response)}"
        try:
            result.raw = response.json()  # show the error object in the raw panel too
        except ValueError:
            pass
        yield result
        return

    if not streaming:
        try:
            data = response.json()
        except ValueError:
            result.ok = False
            result.error = "The server replied, but the reply is not JSON."
            yield result
            return
        read_reply(result, data)
        yield result
        return

    # Streaming: the body is lines like  data: {...}  and ends with  data: [DONE]
    response.encoding = "utf-8"
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue  # blank lines separate events
        result.chunks.append(line)
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            read_chunk(result, json.loads(data))
        except ValueError:
            result.ok = False
            result.error = f"Could not read this streamed chunk as JSON: {data[:200]}"
            yield result
            return
        yield result
    yield result


def send_request(url, payload, api_key=None, timeout=defaults.REQUEST_TIMEOUT):
    """Send the request and return the final Result (no live updates)."""
    result = Result()
    for result in send_request_live(url, payload, api_key=api_key, timeout=timeout):
        pass
    return result


def format_summary(result):
    """A short Markdown summary of a Result, shown in the response panel."""
    if not result.ok:
        return f"### ❌ Error\n\n{result.error}"

    header = f"**HTTP {result.status_code}**"
    if result.chunks:
        header += f" · {len(result.chunks)} chunks"
    parts = [header]

    for choice in result.choices:
        if len(result.choices) > 1:
            parts.append(f"### Choice {choice['index']}")
        else:
            parts.append("### Answer")
        parts.append(f"🤖 {choice['content']}" if choice["content"] else "_(no text)_")
        if choice["reasoning"]:
            parts.append(f"**Reasoning:**\n\n{choice['reasoning']}")
        if choice["refusal"]:
            parts.append(f"**Refusal:** {choice['refusal']}")
        for call in choice["tool_calls"]:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            parts.append(f"🛠️ tool call: `{function.get('name')}({arguments})` id = `{call.get('id')}`")
        parts.append(f"**finish_reason:** `{choice['finish_reason']}`")

    if result.usage:
        usage = result.usage
        parts.append(
            f"**usage:** prompt_tokens = {usage.get('prompt_tokens')}, "
            f"completion_tokens = {usage.get('completion_tokens')}, "
            f"total_tokens = {usage.get('total_tokens')}"
        )
    else:
        parts.append("**usage:** _not reported (when streaming, tick stream_options.include_usage)_")
    return "\n\n".join(parts)
