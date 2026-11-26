"""The Gradio app: 7 tabs of controls on top, request and response side by side below.

It merges the notebooks 7-A-openai-api.ipynb (tabs 1-4) and 7-B-openai-api.ipynb (tabs 5-7).

How it works:
  1. Every control has a short explanation under it.
  2. Optional fields have a "Send ..." checkbox. Unchecked = left out of the JSON.
  3. When any control changes, the request JSON is rebuilt (request_builder.py).
  4. You may edit the JSON by hand. "Send" posts exactly what is in the box (client.py).
"""

import json

import gradio as gr

from . import defaults
from .client import format_summary, send_request_live
from .request_builder import build_request, parse_request_json, to_json

MAX_CHUNKS_SHOWN = 40  # long streams are cut so the page stays fast


def send_box(name, value=False):
    """A small 'Send this field' checkbox placed next to an optional control."""
    return gr.Checkbox(value, label=f"Send {name}", scale=0, min_width=160)


def show_chunks(chunks):
    """The raw SSE lines, cut to the first MAX_CHUNKS_SHOWN."""
    shown = chunks[:MAX_CHUNKS_SHOWN]
    if len(chunks) > MAX_CHUNKS_SHOWN:
        shown.append(f"... ({len(chunks) - MAX_CHUNKS_SHOWN} more chunks)")
    return "\n".join(shown)


def create_app(url=defaults.DEFAULT_URL, model=defaults.DEFAULT_MODEL, api_key=None):
    """Build the gr.Blocks app. The API key stays on the server and is never shown."""

    model_choices = list(defaults.MODEL_CHOICES)
    if model not in model_choices:
        model_choices.insert(0, model)

    with gr.Blocks(title="llama-mate · OpenAI Request Playground") as app:
        gr.Markdown(
            "# 🦙 llama-mate · OpenAI Request Playground\n"
            "Change the controls in the tabs and watch the **request JSON** at the bottom change. "
            "Then press **Send** and read the response next to it.\n\n"
            "💡 Optional fields have a **Send** checkbox. Unchecked = the field is left out of the JSON, "
            "and the server uses its default. ⚠️ Not every server supports every field: if a field has "
            "no effect or the server returns an error, compare what you sent with what came back!"
        )

        # Each control is stored under the same name as the build_request() argument.
        c = {}

        with gr.Tabs():
            # ======================================================= 1 · Core
            with gr.Tab("1 · Core"):
                gr.Markdown(
                    "**`model`** = which brain to use. **`messages`** = the whole conversation so far. "
                    "The model remembers nothing between requests, so the full history is sent every time."
                )
                c["model"] = gr.Dropdown(
                    model_choices, value=model, allow_custom_value=True, label="model",
                    info="Any model you have pulled (`ollama list`). Tools need a model with the 'tools' capability.",
                )
                c["system_role"] = gr.Radio(
                    ["system", "developer"], value="system", label="System message role",
                    info="The boss's instructions. Newer reasoning models call it `developer`.",
                )
                c["system_text"] = gr.Textbox(
                    defaults.DEFAULT_SYSTEM, lines=2, label="system content",
                    info="Leave empty to send no system message",
                )
                c["user_text"] = gr.Textbox(
                    defaults.DEFAULT_USER, lines=2, label="user content", info="The human's question",
                )
                with gr.Row():
                    c["image_url"] = gr.Textbox(
                        "", label="image_url (optional)",
                        info="A link, or a data:image/png;base64,... URL. Turns `content` into a list of blocks.",
                    )
                    c["image_detail"] = gr.Radio(
                        ["auto", "low", "high"], value="auto", label="detail",
                        info="How closely to look at the image",
                    )
                c["add_tool_turn"] = gr.Checkbox(
                    False, label="Add an example tool-call turn",
                    info="Adds an `assistant` message with `tool_calls` and the `tool` message "
                         "that answers it (matched by `tool_call_id`)",
                )

            # =================================================== 2 · Sampling
            with gr.Tab("2 · Sampling"):
                gr.Markdown(
                    "How the model **picks its words**. It writes one token at a time, "
                    "each time choosing from many possible next tokens."
                )
                with gr.Row():
                    c["send_temperature"] = send_box("temperature", True)
                    c["temperature"] = gr.Slider(
                        0, 2, value=0.7, step=0.1, label="temperature",
                        info="Creativity dial: 0 = predictable, 2 = very random",
                    )
                with gr.Row():
                    c["send_top_p"] = send_box("top_p")
                    c["top_p"] = gr.Slider(
                        0, 1, value=1.0, step=0.05, label="top_p",
                        info="Only pick from the most likely words whose chances add up to this. "
                             "Change this OR temperature, not both.",
                    )
                with gr.Row():
                    c["send_max_completion_tokens"] = send_box("max_completion_tokens")
                    c["max_completion_tokens"] = gr.Number(
                        100, precision=0, label="max_completion_tokens",
                        info="Length limit. When hit, the answer is cut off and finish_reason = \"length\".",
                    )
                with gr.Row():
                    c["send_stop"] = send_box("stop")
                    c["stop"] = gr.Textbox(
                        "4.", lines=2, label="stop (one per line, up to 4)",
                        info="'Cut here' markers: the model stops as soon as it writes one. "
                             "Try 'List 10 fruits, numbered.' with stop '4.'",
                    )
                with gr.Row():
                    c["send_seed"] = send_box("seed")
                    c["seed"] = gr.Number(
                        42, precision=0, label="seed",
                        info="Same seed + same request -> (usually) the same answer",
                    )
                with gr.Row():
                    c["send_n"] = send_box("n")
                    c["n"] = gr.Slider(
                        1, 4, value=1, step=1, label="n",
                        info="How many different answers (choices) to generate",
                    )

            # ================================================== 3 · Penalties
            with gr.Tab("3 · Penalties"):
                gr.Markdown("Nudge the model **toward or away from** certain words. Usually left at 0.")
                with gr.Row():
                    c["send_presence_penalty"] = send_box("presence_penalty")
                    c["presence_penalty"] = gr.Slider(
                        -2, 2, value=0, step=0.1, label="presence_penalty",
                        info="Talk about something NEW: penalizes any word that has appeared at least once",
                    )
                with gr.Row():
                    c["send_frequency_penalty"] = send_box("frequency_penalty")
                    c["frequency_penalty"] = gr.Slider(
                        -2, 2, value=0, step=0.1, label="frequency_penalty",
                        info="Don't REPEAT yourself: the penalty grows each time a word is reused",
                    )
                with gr.Row():
                    c["send_logit_bias"] = send_box("logit_bias")
                    c["logit_bias"] = gr.Textbox(
                        defaults.DEFAULT_LOGIT_BIAS, label="logit_bias (JSON: token ID -> -100..100)",
                        info="-100 = never use this token, +100 = strongly prefer it. Token IDs depend on the model.",
                    )

            # ================================================== 4 · Streaming
            with gr.Tab("4 · Streaming"):
                gr.Markdown(
                    "Get the answer **all at once**, or **word by word** as it's written. "
                    "Open **Raw SSE chunks** in the response panel to find `delta` and `[DONE]`."
                )
                c["stream"] = gr.Checkbox(
                    False, label="stream",
                    info="true = Server-Sent Events: many small `data:` chunks instead of one JSON",
                )
                c["include_usage"] = gr.Checkbox(
                    True, label="stream_options.include_usage",
                    info="When streaming: also send the token count in the last chunk",
                )
                c["logprobs"] = gr.Checkbox(
                    False, label="logprobs",
                    info="Also return how confident the model was in each token",
                )
                with gr.Row():
                    c["send_top_logprobs"] = send_box("top_logprobs")
                    c["top_logprobs"] = gr.Slider(
                        0, 20, value=3, step=1, label="top_logprobs",
                        info="Show the top N runner-up tokens at each position (needs logprobs)",
                    )

            # ========================================== 5 · Structured output
            with gr.Tab("5 · Structured output"):
                gr.Markdown(
                    "Force the answer into a **fixed shape**, so code can read it reliably. "
                    "The schema is the same JSON Schema that Pydantic makes with `model_json_schema()`. "
                    "Try asking *\"Weather report: Boston is 72°F.\"*"
                )
                c["response_format"] = gr.Radio(
                    ["text", "json_object", "json_schema"], value="text", label="response_format.type",
                    info="text = normal answer (default, not sent) · json_object = any valid JSON · "
                         "json_schema = JSON matching YOUR template",
                )
                with gr.Row():
                    c["schema_name"] = gr.Textbox(
                        defaults.SCHEMA_NAME, label="json_schema.name", info="A name for your template",
                    )
                    c["strict"] = gr.Checkbox(
                        True, label="strict", info="true = the answer MUST match the schema exactly",
                    )
                c["json_schema"] = gr.Code(
                    json.dumps(defaults.DEFAULT_SCHEMA, indent=2), language="json",
                    label="json_schema.schema (edit me!)", interactive=True,
                )

            # ===================================================== 6 · Tools
            with gr.Tab("6 · Tools"):
                gr.Markdown(
                    "Tell the model **which tools exist**. The model never runs a tool: it only *asks*, "
                    "with `tool_calls` in the response and `finish_reason = \"tool_calls\"`. Your agent runs it. "
                    "Try asking *\"What's the weather in Boston right now?\"*"
                )
                c["send_tools"] = gr.Checkbox(
                    False, label="Send tools",
                    info="The toolbox: a list of tools the model may ask for",
                )
                c["tools"] = gr.Code(
                    json.dumps(defaults.DEFAULT_TOOLS, indent=2), language="json",
                    label="tools (edit me! The description tells the model WHEN to use the tool)",
                    interactive=True,
                )
                with gr.Row():
                    c["tool_choice"] = gr.Radio(
                        ["auto", "none", "required", "specific tool"], value="auto", label="tool_choice",
                        info="auto = model decides · none = never · required = must use one · "
                             "specific = force one tool",
                    )
                    c["forced_tool"] = gr.Textbox(
                        "get_weather", label="specific tool name", info="Used only with 'specific tool'",
                    )
                with gr.Row():
                    c["send_parallel_tool_calls"] = send_box("parallel_tool_calls")
                    c["parallel_tool_calls"] = gr.Checkbox(
                        True, label="parallel_tool_calls",
                        info="Allow several tool calls in one reply (e.g. weather for 3 cities)",
                    )

            # ================================================= 7 · Reasoning
            with gr.Tab("7 · Reasoning"):
                gr.Markdown(
                    "For **thinking** models (e.g. `gpt-oss`): how hard to think before answering. "
                    "Plus an ID for your app's end user."
                )
                with gr.Row():
                    c["send_reasoning_effort"] = send_box("reasoning_effort")
                    c["reasoning_effort"] = gr.Radio(
                        ["low", "medium", "high"], value="medium", label="reasoning_effort",
                        info="low = fast and cheap · high = slower, better on hard problems",
                    )
                with gr.Row():
                    c["send_user_id"] = send_box("user")
                    c["user_id"] = gr.Textbox(
                        defaults.DEFAULT_USER_ID, label="user",
                        info="An ID (not a name) for the end user, for abuse monitoring and rate limits",
                    )

        # ========================================= request | response (bottom)
        with gr.Row(equal_height=False):
            with gr.Column():
                request_box = gr.Code(
                    language="json", label="📤 Request JSON (sent to the LLM server)",
                    interactive=True, lines=20,
                )
                build_status = gr.Markdown(
                    "✏️ You can also edit the JSON by hand before sending. Changing a control rebuilds it."
                )
                url_box = gr.Textbox(url, label="POST to")
                gr.Markdown(f"API key: {'set ✅' if api_key else 'not set (fine for Ollama)'}")
                with gr.Row():
                    send_button = gr.Button("🚀 Send", variant="primary")
                    rebuild_button = gr.Button("Rebuild from controls")
            with gr.Column():
                summary_box = gr.Markdown("📥 _Press **Send** to see the response summary here._")
                raw_box = gr.Code(language="json", label="📥 Raw response", interactive=False)
                chunks_box = gr.Code(label="📥 Raw SSE chunks (when streaming)", interactive=False)

        # ===================================================== wiring
        names = list(c)
        controls = [c[name] for name in names]

        def rebuild(*values):
            """Controls -> request JSON. Shows a warning instead if a JSON box is invalid."""
            try:
                request = build_request(**dict(zip(names, values)))
            except ValueError as error:
                return gr.skip(), f"⚠️ {error}"
            return to_json(request), "✏️ You can also edit the JSON by hand before sending. Changing a control rebuilds it."

        def send(url_text, request_text):
            """Post the request box and stream results into the response panel."""
            try:
                payload = parse_request_json(request_text)
            except ValueError as error:
                yield f"### ❌ Invalid request JSON\n\n{error}", "", ""
                return
            yield "_Waiting for the server..._", "", ""
            for result in send_request_live(url_text, payload, api_key=api_key):
                raw = to_json(result.raw) if result.raw else ""
                yield format_summary(result), raw, show_chunks(result.chunks)

        # Every control rebuilds the JSON, so it updates live.
        gr.on(
            triggers=[control.change for control in controls] + [rebuild_button.click, app.load],
            fn=rebuild,
            inputs=controls,
            outputs=[request_box, build_status],
        )
        send_button.click(
            send, inputs=[url_box, request_box], outputs=[summary_box, raw_box, chunks_box]
        )

    return app
