"""Turn the values of the UI controls into a Chat Completions request (a Python dict).

There is no Gradio in this file, so it is easy to test and to reuse.
The pattern used everywhere: an optional field is only added when its "Send" box is ticked.
"""

import copy
import json

from . import defaults


def parse_json_field(text, what):
    """Parse JSON typed by the student, with an error message they can understand."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{what} is not valid JSON: {error.msg} "
            f"(line {error.lineno}, column {error.colno})"
        ) from None


def parse_stop(text):
    """One stop text per line, at most 4. Type \\n to mean a real newline."""
    stops = [line.replace("\\n", "\n") for line in text.splitlines() if line.strip()]
    return stops[:4]


def build_messages(system_role, system_text, user_text, image_url, image_detail, add_tool_turn):
    """Tab 1: the conversation so far."""
    messages = []
    if system_text.strip():
        messages.append({"role": system_role, "content": system_text})

    if image_url.strip():
        # Text + image: content becomes a LIST of blocks.
        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url.strip(), "detail": image_detail}},
        ]
    else:
        # Text only: content is a plain string.
        content = user_text
    messages.append({"role": "user", "content": content})

    if add_tool_turn:
        # An earlier turn: the model asked for a tool, and the agent answered.
        messages += copy.deepcopy(defaults.EXAMPLE_TOOL_TURN)
    return messages


def build_request(
    # ---------- 1. Core
    model=defaults.DEFAULT_MODEL,
    system_role="system",
    system_text=defaults.DEFAULT_SYSTEM,
    user_text=defaults.DEFAULT_USER,
    image_url="",
    image_detail="auto",
    add_tool_turn=False,
    # ---------- 2. Sampling
    send_temperature=True,
    temperature=0.7,
    send_top_p=False,
    top_p=1.0,
    send_max_completion_tokens=False,
    max_completion_tokens=100,
    send_stop=False,
    stop="4.",
    send_seed=False,
    seed=42,
    send_n=False,
    n=1,
    # ---------- 3. Penalties & token bias
    send_presence_penalty=False,
    presence_penalty=0.0,
    send_frequency_penalty=False,
    frequency_penalty=0.0,
    send_logit_bias=False,
    logit_bias=defaults.DEFAULT_LOGIT_BIAS,
    # ---------- 4. Streaming & logprobs
    stream=False,
    include_usage=True,
    logprobs=False,
    send_top_logprobs=False,
    top_logprobs=3,
    # ---------- 5. Structured output ("text", "json_object" or "json_schema")
    response_format="text",
    schema_name=defaults.SCHEMA_NAME,
    strict=True,
    json_schema=json.dumps(defaults.DEFAULT_SCHEMA, indent=2),
    # ---------- 6. Tools
    send_tools=False,
    tools=json.dumps(defaults.DEFAULT_TOOLS, indent=2),
    tool_choice="auto",  # "auto", "none", "required" or "specific tool"
    forced_tool="get_weather",
    send_parallel_tool_calls=False,
    parallel_tool_calls=True,
    # ---------- 7. Reasoning & metadata
    send_reasoning_effort=False,
    reasoning_effort="medium",
    send_user_id=False,
    user_id=defaults.DEFAULT_USER_ID,
):
    """Build the request dict. Raises ValueError if a JSON text box is invalid."""

    # ---------- 1. Core: model + messages are the only required fields
    messages = build_messages(system_role, system_text, user_text, image_url, image_detail, add_tool_turn)
    body = {"model": model, "messages": messages}

    # ---------- 2. Sampling
    if send_temperature:
        body["temperature"] = round(float(temperature), 2)
    if send_top_p:
        body["top_p"] = round(float(top_p), 2)
    if send_max_completion_tokens:
        body["max_completion_tokens"] = int(max_completion_tokens)
    if send_stop:
        stops = parse_stop(stop)
        if stops:
            body["stop"] = stops
    if send_seed:
        body["seed"] = int(seed)
    if send_n:
        body["n"] = int(n)

    # ---------- 3. Penalties & token bias
    if send_presence_penalty:
        body["presence_penalty"] = round(float(presence_penalty), 2)
    if send_frequency_penalty:
        body["frequency_penalty"] = round(float(frequency_penalty), 2)
    if send_logit_bias:
        bias = parse_json_field(logit_bias or "{}", "logit_bias")
        if not isinstance(bias, dict):
            raise ValueError('logit_bias must be a JSON object, e.g. {"50256": -100}')
        body["logit_bias"] = bias

    # ---------- 4. Streaming & logprobs
    body["stream"] = bool(stream)  # always sent, so you can see true vs false
    if stream and include_usage:
        body["stream_options"] = {"include_usage": True}
    if logprobs:
        body["logprobs"] = True
        if send_top_logprobs:
            body["top_logprobs"] = int(top_logprobs)

    # ---------- 5. Structured output
    if response_format == "json_object":
        body["response_format"] = {"type": "json_object"}
    elif response_format == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": bool(strict),
                "schema": parse_json_field(json_schema, "The schema"),
            },
        }
    # "text" -> left out: plain text is the default

    # ---------- 6. Tools
    if send_tools:
        tool_list = parse_json_field(tools, "The tools list")
        if not isinstance(tool_list, list):
            raise ValueError("The tools list must be a JSON list: [ {...}, ... ]")
        body["tools"] = tool_list
        if tool_choice == "specific tool":
            body["tool_choice"] = {"type": "function", "function": {"name": forced_tool}}
        else:
            body["tool_choice"] = tool_choice
        if send_parallel_tool_calls:
            body["parallel_tool_calls"] = bool(parallel_tool_calls)

    # ---------- 7. Reasoning & metadata
    if send_reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if send_user_id:
        body["user"] = user_id

    return body


def to_json(request):
    """Pretty JSON text for the request box."""
    return json.dumps(request, indent=2, ensure_ascii=False)


def parse_request_json(text):
    """Read the (possibly hand-edited) request box back into a dict."""
    request = parse_json_field(text, "The request")
    if not isinstance(request, dict):
        raise ValueError("The request must be a JSON object: { ... }")
    return request
