"""Default values used by the app. Change them here, not inside the UI code.

They come from the notebooks 7-A-openai-api.ipynb and 7-B-openai-api.ipynb.
"""

# Ollama serves an OpenAI-compatible API at this address.
DEFAULT_URL = "http://localhost:11434/v1/chat/completions"

# A small model that runs on most laptops. Get it with: ollama pull llama3.2:3b
DEFAULT_MODEL = "llama3.2:3b"

# Suggestions shown in the model dropdown (you can also type any other name).
MODEL_CHOICES = ["llama3.2:3b", "qwen3:1.7b", "qwen3:8b", "llama3.1:8b", "gpt-oss:20b"]

DEFAULT_SYSTEM = "You are a helpful assistant. Answer in 2 sentences."
DEFAULT_USER = "Explain what an AI agent is."

# How long (seconds) to wait for the server before giving up.
REQUEST_TIMEOUT = 300

# --- Core (tab 1): an earlier turn where the model asked for a tool --------

EXAMPLE_TOOL_TURN = [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"location\": \"Boston, MA\"}"},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "{\"temperature\": \"72F\", \"condition\": \"Sunny\"}",
    },
]

# --- Penalties (tab 3) -----------------------------------------------------

# Token ID -> bias. Token IDs depend on the model's tokenizer.
DEFAULT_LOGIT_BIAS = '{"50256": -100}'

# --- Structured output (tab 5) ---------------------------------------------

SCHEMA_NAME = "weather_report"

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temperature_f": {"type": "number"},
    },
    "required": ["city", "temperature_f"],
    "additionalProperties": False,
}

# --- Tools (tab 6) ---------------------------------------------------------

# The model never runs this tool: it only asks us to run it.
DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }
]

# --- Reasoning & metadata (tab 7) ------------------------------------------

DEFAULT_USER_ID = "user_98765"
