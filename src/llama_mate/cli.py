"""Command line entry point: `llama-mate --help`."""

import argparse
import os

from . import __version__, defaults


def parse_args(argv=None):
    """Read the command line flags. Most fall back to an environment variable."""
    parser = argparse.ArgumentParser(
        prog="llama-mate",
        description="A playground for learning the OpenAI Chat Completions request.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("LLAMA_MATE_URL", defaults.DEFAULT_URL),
        help="Chat Completions endpoint (env: LLAMA_MATE_URL, default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLAMA_MATE_MODEL", defaults.DEFAULT_MODEL),
        help="Model name to start with (env: LLAMA_MATE_MODEL, default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for cloud servers, sent as 'Authorization: Bearer ...' (env: OPENAI_API_KEY)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: %(default)s)")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: %(default)s)")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link (off by default)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser window"
    )
    parser.add_argument("--version", action="version", version=f"llama-mate {__version__}")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Imported here so that `llama-mate --help` stays fast.
    from .app import create_app

    app = create_app(url=args.url, model=args.model, api_key=args.api_key)
    # Note: we never print the API key, only whether one is set.
    print(f"llama-mate {__version__}: server {args.url}, model {args.model}, "
          f"API key {'set' if args.api_key else 'not set'}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )
