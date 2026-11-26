# 🦙 llama-mate

Demystify the LLM communication.

**llama-mate** is a small web app for learning the **OpenAI Chat Completions API**, the
request format used by OpenAI, Ollama, vLLM, LM Studio, llama.cpp and many other LLM servers.

You change a control, you see the request JSON change, you press **Send**, and you read the
response right next to it. Every control explains what its field does.

```
┌───────────────────────────────────────────────────────────────┐
│ 1. Core │ 2. Sampling │ 3. Penalties │ 4. Streaming │ ...     │  ← one tab per topic
│   controls, each with an explanation and a "Send …" checkbox  │
├──────────────────────────────┬────────────────────────────────┤
│ Request JSON (live, editable)│ Response: answer, finish_reason│
│ [Send]                       │ tool calls, usage, raw SSE     │
└──────────────────────────────┴────────────────────────────────┘
```

## What you will learn

| Tab | Fields |
|-----|--------|
| 1 · Core | `model`, `messages` (system/developer role, text, image, an example tool-call turn) |
| 2 · Sampling | `temperature`, `top_p`, `max_completion_tokens`, `stop`, `seed`, `n` |
| 3 · Penalties | `presence_penalty`, `frequency_penalty`, `logit_bias` |
| 4 · Streaming | `stream`, `stream_options.include_usage`, `logprobs`, `top_logprobs` |
| 5 · Structured output | `response_format` (`text`, `json_object`, `json_schema` with `name`, `strict`, `schema`) |
| 6 · Tools | `tools`, `tool_choice` (incl. forcing one tool), `parallel_tool_calls` |
| 7 · Reasoning | `reasoning_effort`, `user` |

Optional fields have a **Send …** checkbox. When it is unchecked, the field is left out of
the JSON, so the server uses its own default. That shows you which fields are required
(only `model` and `messages`) and which are optional.

---

## 1. Install

You need **Python 3.10 or newer**. Check with:

```bash
python3 --version
```

We recommend a virtual environment, so llama-mate does not mix with your other projects:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install git+https://github.com/ihawke/llama-mate
```

### Install a specific version

Your instructor may ask everyone to use the same version. Add `@` and the tag:

```bash
pip install git+https://github.com/ihawke/llama-mate@v0.1.0
```

To upgrade later to the newest version:

```bash
pip install --upgrade git+https://github.com/ihawke/llama-mate
```

## 2. Start an LLM server (Ollama)

llama-mate talks to **Ollama** by default.

1. Install Ollama from <https://ollama.com/download>.
2. Download the default model (about 2 GB):

   ```bash
   ollama pull llama3.2:3b
   ```

3. Make sure Ollama is running: open the Ollama app, or run `ollama serve` in a terminal.

For the **Tools** tab you need a model with the *tools* capability (check with `ollama show MODEL`).
For the **Reasoning** tab, pull a thinking model, for example `ollama pull gpt-oss:20b`.

## 3. Run llama-mate

```bash
llama-mate
```

Your browser opens at <http://127.0.0.1:7860>. If it does not, open that address yourself.
Stop the app with **Ctrl + C** in the terminal.

You can also start it with `python -m llama_mate`.

## 4. Try it

**Tabs 1–4**

1. **Tab 2:** set `temperature` to 0 and press Send 3 times. Then set it to 1.8. How different are the answers?
2. **Tab 2:** ask `List 10 fruits, numbered.`, tick **Send stop** with `4.`. Where does the answer end? What is `finish_reason`?
3. **Tab 2:** tick **Send max_completion_tokens** with `10`. What is `finish_reason` now?
4. **Tab 4:** turn on `stream`. Compare the raw SSE chunks with the non-streamed response. Find `delta` and `[DONE]`.
5. **Tab 1:** turn on the example tool-call turn and ask *"What should I wear today?"*. Does the model use the tool result (72°F, sunny)?
6. **Tab 2:** set `n` to 3. How many `choices` came back? (If only one, your server ignores `n`!)

**Tabs 5–7** (start with the user content *"What's the weather in Boston right now?"*)

7. **Tab 6:** tick **Send tools** and send. Did the model answer with text, or with a 🛠️ tool call? Check `finish_reason`.
8. **Tab 6:** set `tool_choice` to `none`, then `required`. What changes?
9. **Tab 6:** ask *"What's the weather in Boston, Paris and Tokyo?"*. How many tool calls come back?
10. **Tab 6:** rewrite the tool's `description` badly (e.g. `"Does stuff."`). Does the model still use it correctly?
11. **Tab 5:** untick **Send tools**, choose `json_schema` and ask *"Weather report: Boston is 72°F."*. Is the answer exactly `{"city": ..., "temperature_f": ...}`?
12. **Tab 5:** add a `"condition"` field to the schema (and to `required`). Does the answer follow your new template?
13. **Tab 7:** with a thinking model (e.g. `gpt-oss:20b`), compare `reasoning_effort` `low` vs `high` on a tricky math question. Watch `usage`.

**Edit by hand:** change the request JSON directly, then press **Send**. Break it on purpose
(delete a `}`) to see the error message. **Rebuild from controls** puts it back.

---

## Command line options

| Flag | Environment variable | Default | What it does |
|------|----------------------|---------|--------------|
| `--url` | `LLAMA_MATE_URL` | `http://localhost:11434/v1/chat/completions` | Where requests are sent |
| `--model` | `LLAMA_MATE_MODEL` | `llama3.2:3b` | Model name the app starts with |
| `--api-key` | `OPENAI_API_KEY` | none | Sent as `Authorization: Bearer ...` (cloud servers) |
| `--host` | | `127.0.0.1` | Address the app listens on (only your computer) |
| `--port` | | `7860` | Port the app listens on |
| `--share` | | off | Make a temporary public Gradio link |
| `--no-browser` | | off | Do not open a browser window |

A flag wins over its environment variable. See all options with `llama-mate --help`.

Examples:

```bash
llama-mate --model qwen3:8b
llama-mate --port 8000 --no-browser
```

## Using other servers

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint works.

**Ollama on another computer:**

```bash
llama-mate --url http://192.168.1.50:11434/v1/chat/completions
```

**OpenAI (cloud):** put your key in an environment variable, **not** on the command line
(commands are saved in your shell history):

```bash
export OPENAI_API_KEY="sk-..."      # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
llama-mate --url https://api.openai.com/v1/chat/completions --model gpt-4o-mini
```

The same works for other providers (Groq, OpenRouter, Together, ...): use their
chat-completions URL, their model name, and their key in `OPENAI_API_KEY`.

> **Safety notes**
> - The key stays on your computer. llama-mate never shows it in the page or prints it.
> - Never commit a key to Git or paste it into the request JSON.
> - `--share` makes the app reachable from the internet, and anyone with the link uses
>   *your* key and server. Only use it when you need it.

## Troubleshooting

| Message | What to do |
|---------|------------|
| *Could not reach the server* | Ollama is not running. Start the Ollama app or run `ollama serve`. Check `--url`. |
| *HTTP 404 … model not found* | Pull the model (`ollama pull llama3.2:3b`) or change `model` to one from `ollama list`. |
| *HTTP 401* | The API key is missing or wrong (cloud servers). |
| *HTTP 400* | The server does not accept a field. Untick it, or check the field's value. |
| *... does not support tools* | Pick a model with the tools capability (`ollama show MODEL`). |
| *Invalid request JSON* | You edited the JSON by hand and broke it. Fix it, or press **Rebuild from controls**. |
| `llama-mate: command not found` | Activate your virtual environment, or run `python -m llama_mate`. |
| Port already in use | Use another port: `llama-mate --port 7861`. |

Not every server supports every field. Ollama may ignore some of them (for example `n` or
`logit_bias`). If a field has no effect, or the server returns an error, that's part of the
lesson: compare the JSON you sent with what came back!

---

## For developers

```bash
git clone https://github.com/ihawke/llama-mate
cd llama-mate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

The tests use a small fake server, so you do not need Ollama to run them.

```
src/llama_mate/
    cli.py               # command line flags, then starts the app
    app.py               # the Gradio UI: 7 tabs, request and response side by side
    request_builder.py   # controls -> request dict (no Gradio, easy to test)
    client.py            # sends the request: normal replies, streaming (SSE), errors
    defaults.py          # default model, URL, JSON schema and tool definition
tests/                   # pytest tests
```

To release a version students can pin: update `__version__` in
`src/llama_mate/__init__.py`, commit, then `git tag v0.1.0 && git push --tags`.
