"""Command line flags and their environment-variable fallbacks."""

from llama_mate import defaults
from llama_mate.cli import parse_args


def test_defaults(monkeypatch):
    for name in ["LLAMA_MATE_URL", "LLAMA_MATE_MODEL", "OPENAI_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    args = parse_args([])
    assert args.url == defaults.DEFAULT_URL
    assert args.model == defaults.DEFAULT_MODEL
    assert args.api_key is None
    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.share is False
    assert args.no_browser is False


def test_environment_fallbacks(monkeypatch):
    monkeypatch.setenv("LLAMA_MATE_URL", "http://example/v1/chat/completions")
    monkeypatch.setenv("LLAMA_MATE_MODEL", "qwen3:4b")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    args = parse_args([])
    assert args.url == "http://example/v1/chat/completions"
    assert args.model == "qwen3:4b"
    assert args.api_key == "sk-env"


def test_flags_beat_environment(monkeypatch):
    monkeypatch.setenv("LLAMA_MATE_MODEL", "from-env")
    args = parse_args(["--model", "from-flag", "--share", "--no-browser", "--port", "8000"])
    assert args.model == "from-flag"
    assert args.share and args.no_browser and args.port == 8000
