"""LLM provider chain: claude -p CLI (Max subscription, $0) -> Gemini free tier
-> anthropic SDK (only if ANTHROPIC_API_KEY set). Every call returns a uniform
LLMResponse so token accounting works regardless of backend.

Env: LLM_BACKEND=auto|cli|gemini|api (default auto), CLAUDE_MODEL=haiku|sonnet
(default haiku), GEMINI_MODEL (default gemini-2.0-flash), ANTHROPIC_API_KEY,
GEMINI_API_KEY, SPACE_ID (set automatically by HF Spaces -- used to skip the
CLI backend, which isn't installed there).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv()


class ConfigError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    backend: str
    model: str
    latency_s: float
    cached: bool = False


_CLI_MODEL_MAP = {"fast": "haiku", "smart": os.environ.get("CLAUDE_MODEL", "sonnet")}
_GEMINI_MODEL_MAP = {"fast": "gemini-2.0-flash", "smart": "gemini-2.0-flash"}
_API_MODEL_MAP = {"fast": "claude-haiku-4-5-20251001", "smart": "claude-sonnet-5"}


def _resolve_backend() -> str:
    forced = os.environ.get("LLM_BACKEND", "auto")
    if forced != "auto":
        return forced
    on_space = bool(os.environ.get("SPACE_ID"))
    if not on_space and shutil.which("claude"):
        return "cli"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    raise ConfigError(
        "No LLM backend available: no `claude` CLI on PATH, no GEMINI_API_KEY, "
        "no ANTHROPIC_API_KEY. Set one of these to proceed."
    )


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _generate_cli(prompt: str, system: str | None, tier: str, max_tokens: int,
                   json_only: bool, timeout_s: int) -> LLMResponse:
    model = _CLI_MODEL_MAP[tier]
    full_prompt = prompt
    if json_only:
        full_prompt += "\n\nRespond with ONLY valid JSON, no prose, no code fences."
    args = ["claude", "-p", "--model", model, "--output-format", "json"]
    if system:
        args += ["--append-system-prompt", system]

    start = time.time()
    for attempt in range(2):
        try:
            proc = subprocess.run(
                args, input=full_prompt, capture_output=True, text=True,
                encoding="utf-8", timeout=timeout_s,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            break
        except subprocess.TimeoutExpired:
            if attempt == 1:
                raise
            continue
    latency = time.time() - start

    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed (exit {proc.returncode}): {proc.stderr[:500]}")

    raw = proc.stdout.strip()
    input_tok = output_tok = None
    text = raw
    try:
        parsed = json.loads(raw)
        text = parsed.get("result", raw)
        usage = parsed.get("usage", {})
        input_tok = usage.get("input_tokens")
        output_tok = usage.get("output_tokens")
    except (json.JSONDecodeError, AttributeError):
        pass  # fall back to plain text, token counts stay None

    if json_only:
        text = _strip_json_fences(text)

    return LLMResponse(text=text, input_tokens=input_tok, output_tokens=output_tok,
                        backend="cli", model=model, latency_s=latency)


def _generate_gemini(prompt: str, system: str | None, tier: str, max_tokens: int,
                      temperature: float, json_only: bool, timeout_s: int) -> LLMResponse:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", _GEMINI_MODEL_MAP[tier])
    config = {"temperature": temperature, "max_output_tokens": max_tokens}
    if json_only:
        config["response_mime_type"] = "application/json"
    if system:
        config["system_instruction"] = system

    start = time.time()
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    latency = time.time() - start

    usage = getattr(resp, "usage_metadata", None)
    input_tok = getattr(usage, "prompt_token_count", None) if usage else None
    output_tok = getattr(usage, "candidates_token_count", None) if usage else None
    text = _strip_json_fences(resp.text) if json_only else resp.text

    return LLMResponse(text=text, input_tokens=input_tok, output_tokens=output_tok,
                        backend="gemini", model=model, latency_s=latency)


def _generate_api(prompt: str, system: str | None, tier: str, max_tokens: int,
                   temperature: float, json_only: bool, timeout_s: int) -> LLMResponse:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = _API_MODEL_MAP[tier]
    full_prompt = prompt + ("\n\nRespond with ONLY valid JSON." if json_only else "")

    start = time.time()
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system or "", messages=[{"role": "user", "content": full_prompt}],
    )
    latency = time.time() - start
    text = resp.content[0].text
    if json_only:
        text = _strip_json_fences(text)

    return LLMResponse(text=text, input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens,
                        backend="api", model=model, latency_s=latency)


def generate(prompt: str, *, system: str | None = None,
             tier: Literal["fast", "smart"] = "fast",
             max_tokens: int = 1024, temperature: float = 0.3,
             json_only: bool = False, timeout_s: int = 120) -> LLMResponse:
    backend = _resolve_backend()
    if backend == "cli":
        return _generate_cli(prompt, system, tier, max_tokens, json_only, timeout_s)
    if backend == "gemini":
        return _generate_gemini(prompt, system, tier, max_tokens, temperature, json_only, timeout_s)
    if backend == "api":
        return _generate_api(prompt, system, tier, max_tokens, temperature, json_only, timeout_s)
    raise ConfigError(f"Unknown LLM_BACKEND: {backend}")
