"""Inference layer.

Two providers behind one function. Switch with LLM_PROVIDER in .env —
cloud for iteration speed, local for air-gapped OT networks.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")


def _anthropic(system: str, user: str) -> str:
    from anthropic import Anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY missing from .env")

    msg = Anthropic(api_key=key).messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def _ollama(system: str, user: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def ask(system: str, user: str) -> str:
    return _ollama(system, user) if PROVIDER == "ollama" else _anthropic(system, user)
