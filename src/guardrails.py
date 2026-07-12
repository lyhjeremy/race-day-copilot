"""Reusable guardrail patterns: schema-validated generation with a retry loop,
claim grounding, confidence gating, and PII redaction.

This module is where every project's deterministic verifier plugs in via the
`verifier` argument to generate_validated() -- see each project's verifier.py.
"""
from __future__ import annotations

import json
import re
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

import llm

T = TypeVar("T", bound=BaseModel)


class GuardrailError(RuntimeError):
    def __init__(self, message: str, last_output: str, violations: list[str]):
        super().__init__(message)
        self.last_output = last_output
        self.violations = violations


class Refusal(BaseModel):
    reason: str
    user_message: str
    confidence: float = 0.0


def _tolerant_json_parse(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    # strip trailing commas before } or ]
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    return json.loads(t)


def generate_validated(prompt: str, schema: type[T], *,
                        verifier: Callable[[T], list[str]] | None = None,
                        max_retries: int = 2,
                        llm_kwargs: dict | None = None) -> T:
    """Generate JSON matching `schema`, retrying with error feedback on failure.

    verifier(instance) -> list of human-readable violation strings (empty =
    pass). Used for deterministic checks the schema alone can't express
    (arithmetic sums, reconciliation, enum-domain consistency, ...).
    """
    kwargs = {**(llm_kwargs or {}), "json_only": True}
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    working_prompt = (
        f"{prompt}\n\nRespond with JSON matching this schema exactly:\n{schema_hint}"
    )

    last_output = ""
    all_violations: list[str] = []

    for attempt in range(max_retries + 1):
        resp = llm.generate(working_prompt, **kwargs)
        last_output = resp.text
        violations: list[str] = []

        try:
            data = _tolerant_json_parse(resp.text)
            instance = schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            violations = [f"Schema validation failed: {e}"]
        else:
            if verifier:
                violations = verifier(instance)
            if not violations:
                return instance

        all_violations = violations
        if attempt < max_retries:
            working_prompt = (
                f"{prompt}\n\nYour previous output failed these checks:\n"
                + "\n".join(f"- {v}" for v in violations)
                + f"\n\nYour previous output was:\n{resp.text}\n\n"
                "Regenerate the FULL JSON, fixing every issue above. "
                f"Respond with JSON matching this schema exactly:\n{schema_hint}"
            )

    raise GuardrailError(
        f"generate_validated exhausted {max_retries} retries", last_output, all_violations
    )


class GroundingReport(BaseModel):
    grounded: list[str]
    ungrounded: list[str]
    grounding_rate: float


def ground_claims(claims: list[str], sources: list[str], threshold: float = 0.55) -> GroundingReport:
    """Check each claim against a fuzzy match to any source chunk (embedding
    cosine similarity). Claims below threshold are flagged ungrounded.
    """
    if not claims:
        return GroundingReport(grounded=[], ungrounded=[], grounding_rate=1.0)
    if not sources:
        return GroundingReport(grounded=[], ungrounded=list(claims), grounding_rate=0.0)

    from cache import embed
    import numpy as np

    source_embs = np.stack([embed(s) for s in sources])
    grounded, ungrounded = [], []
    for claim in claims:
        claim_emb = embed(claim)
        sims = source_embs @ claim_emb
        if float(sims.max()) >= threshold:
            grounded.append(claim)
        else:
            ungrounded.append(claim)

    rate = len(grounded) / len(claims)
    return GroundingReport(grounded=grounded, ungrounded=ungrounded, grounding_rate=round(rate, 3))


class ConfidenceGate:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def check(self, confidence: float, *, retry_message: str) -> Refusal | None:
        if confidence < self.threshold:
            return Refusal(
                reason=f"confidence {confidence:.2f} below threshold {self.threshold}",
                user_message=retry_message,
                confidence=confidence,
            )
        return None


# --- PII redaction -----------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, int]:
    """Mask card numbers (Luhn-validated only, to last-4), emails, and phone
    numbers. Returns (clean_text, n_redactions).
    """
    n = 0

    def _card_sub(m: re.Match) -> str:
        nonlocal n
        digits = re.sub(r"[ -]", "", m.group(0))
        if len(digits) >= 13 and _luhn_valid(digits):
            n += 1
            return f"[card ending {digits[-4:]}]"
        return m.group(0)

    text = _CARD_RE.sub(_card_sub, text)

    def _email_sub(m: re.Match) -> str:
        nonlocal n
        n += 1
        return "[redacted email]"

    text = _EMAIL_RE.sub(_email_sub, text)

    def _phone_sub(m: re.Match) -> str:
        nonlocal n
        n += 1
        return "[redacted phone]"

    text = _PHONE_RE.sub(_phone_sub, text)

    return text, n
