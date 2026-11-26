"""Builder output for each tab's fields."""

import json

import pytest

from llama_mate.request_builder import build_request, parse_request_json, to_json

# Turn off the one field that is sent by default, to see the bare minimum.
MINIMAL = {"send_temperature": False}


# ---------- 1. Core

def test_core_minimum_is_model_messages_and_stream():
    body = build_request(model="m", system_text="be nice", user_text="hi", **MINIMAL)
    assert body == {
        "model": "m",
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
    }


def test_core_developer_role_and_empty_system():
    body = build_request(system_role="developer", system_text="rules")
    assert body["messages"][0]["role"] == "developer"
    body = build_request(system_text="   ", user_text="hi")
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_core_image_turns_content_into_blocks():
    body = build_request(user_text="What is this?", image_url=" https://x/cat.jpg ", image_detail="low")
    content = body["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://x/cat.jpg", "detail": "low"}},
    ]


def test_core_example_tool_turn():
    body = build_request(add_tool_turn=True)
    assistant, tool = body["messages"][-2:]
    assert assistant["role"] == "assistant" and assistant["content"] is None
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]


# ---------- 2. Sampling

def test_temperature_is_sent_by_default():
    assert build_request()["temperature"] == 0.7


def test_sampling():
    body = build_request(
        temperature=0.30000001,
        send_top_p=True, top_p=0.5,
        send_max_completion_tokens=True, max_completion_tokens=10.0,
        send_stop=True, stop="4.\n\nEND\\n\na\nb\nc",
        send_seed=True, seed=7.0,
        send_n=True, n=3.0,
    )
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.5
    assert body["max_completion_tokens"] == 10
    assert body["stop"] == ["4.", "END\n", "a", "b"]  # \n becomes a newline, max 4
    assert body["seed"] == 7
    assert body["n"] == 3


def test_unticked_fields_are_left_out():
    body = build_request(**MINIMAL)
    for key in ["temperature", "top_p", "max_completion_tokens", "stop", "seed", "n",
                "presence_penalty", "frequency_penalty", "logit_bias", "stream_options",
                "logprobs", "top_logprobs", "response_format", "tools", "tool_choice",
                "parallel_tool_calls", "reasoning_effort", "user"]:
        assert key not in body


def test_empty_stop_is_left_out():
    assert "stop" not in build_request(send_stop=True, stop="\n  \n")


# ---------- 3. Penalties & token bias

def test_penalties_and_logit_bias():
    body = build_request(
        send_presence_penalty=True, presence_penalty=1.5,
        send_frequency_penalty=True, frequency_penalty=-0.5,
        send_logit_bias=True, logit_bias='{"12366": -100}',
    )
    assert body["presence_penalty"] == 1.5
    assert body["frequency_penalty"] == -0.5
    assert body["logit_bias"] == {"12366": -100}


def test_logit_bias_errors():
    with pytest.raises(ValueError, match="logit_bias is not valid JSON"):
        build_request(send_logit_bias=True, logit_bias="{oops")
    with pytest.raises(ValueError, match="JSON object"):
        build_request(send_logit_bias=True, logit_bias="[1]")


# ---------- 4. Streaming & logprobs

def test_streaming_and_usage():
    body = build_request(stream=True, include_usage=True)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    # include_usage only matters when streaming
    assert "stream_options" not in build_request(stream=False, include_usage=True)


def test_logprobs():
    body = build_request(logprobs=True, send_top_logprobs=True, top_logprobs=5)
    assert body["logprobs"] is True and body["top_logprobs"] == 5
    # top_logprobs needs logprobs
    assert "top_logprobs" not in build_request(logprobs=False, send_top_logprobs=True)


# ---------- 5. Structured output

def test_json_object():
    assert build_request(response_format="json_object")["response_format"] == {"type": "json_object"}


def test_json_schema():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    body = build_request(response_format="json_schema", schema_name="my_template",
                         strict=False, json_schema=json.dumps(schema))
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "my_template", "strict": False, "schema": schema},
    }


def test_json_schema_invalid():
    with pytest.raises(ValueError, match="schema is not valid JSON"):
        build_request(response_format="json_schema", json_schema="{oops")


# ---------- 6. Tools

def test_tools_with_choice_and_parallel():
    body = build_request(send_tools=True, tool_choice="required",
                         send_parallel_tool_calls=True, parallel_tool_calls=False)
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False


def test_specific_tool_is_forced():
    body = build_request(send_tools=True, tool_choice="specific tool", forced_tool="get_weather")
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_tool_options_need_tools():
    body = build_request(send_tools=False, tool_choice="required", send_parallel_tool_calls=True)
    assert "tool_choice" not in body and "parallel_tool_calls" not in body


def test_tools_errors():
    with pytest.raises(ValueError, match="tools list is not valid JSON"):
        build_request(send_tools=True, tools="[{")
    with pytest.raises(ValueError, match="must be a JSON list"):
        build_request(send_tools=True, tools='{"type": "function"}')


# ---------- 7. Reasoning & metadata

def test_reasoning_and_user():
    body = build_request(send_reasoning_effort=True, reasoning_effort="high",
                         send_user_id=True, user_id="user_1")
    assert body["reasoning_effort"] == "high"
    assert body["user"] == "user_1"


# ---------- the request box

def test_json_round_trip_and_errors():
    body = build_request(add_tool_turn=True)
    assert parse_request_json(to_json(body)) == body
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_request_json('{"model": ')
    with pytest.raises(ValueError, match="JSON object"):
        parse_request_json("[1, 2]")
