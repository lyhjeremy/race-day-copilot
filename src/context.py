"""Token estimation + context budgeting.

Calibration: chars-per-token constants below are a documented heuristic, not a
measured constant (no live tokenizer available offline). When GEMINI_API_KEY
lands, `calibrate()` measures the real ratio against Gemini's count_tokens and
prints an update suggestion — see toolkit README note.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

# Heuristic chars/token — English prose tokenizers average ~4 chars/token;
# we use a slightly conservative 3.7 to avoid under-budgeting. CJK scripts are
# ~1 token per 1-2 characters in BPE tokenizers; we use 1.9 as a middle estimate.
_EN_CHARS_PER_TOKEN = 3.7
_CJK_CHARS_PER_TOKEN = 1.9
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


def estimate_tokens(text: str) -> int:
    """Estimate token count for `text` using a calibrated chars/token heuristic.

    Splits text into CJK vs non-CJK runs and estimates each separately, since
    a single global ratio badly under/over-counts mixed-language text (e.g.
    Menu Decoder's dish names).
    """
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars
    return round(cjk_chars / _CJK_CHARS_PER_TOKEN + other_chars / _EN_CHARS_PER_TOKEN)


def calibrate(sample_text: str, measured_tokens: int) -> float:
    """Return the actual chars/token ratio for a sample, given a real count
    (e.g. from Gemini's count_tokens). Call once live and compare to the
    constants above; update this module's constants if they're off by >15%.
    """
    if measured_tokens == 0:
        return 0.0
    return len(sample_text) / measured_tokens


@dataclass
class Section:
    name: str
    items: list[str]  # rank-ordered, most important first
    priority: int = 1  # 1 = highest priority for extra budget
    min_tokens: int = 0
    max_tokens: int | None = None
    joiner: str = "\n"


@dataclass
class SectionReport:
    name: str
    items_total: int
    items_kept: int
    tokens: int
    truncated: bool


@dataclass
class PackedPrompt:
    prompt: str
    report: list[SectionReport] = field(default_factory=list)

    def report_markdown(self) -> str:
        lines = ["| Section | Items kept | Tokens |", "|---|---|---|"]
        for r in self.report:
            kept = f"{r.items_kept}/{r.items_total}" + (" (truncated)" if r.truncated else "")
            lines.append(f"| {r.name} | {kept} | {r.tokens} |")
        lines.append(f"| **Total** | | **{sum(r.tokens for r in self.report)}** |")
        return "\n".join(lines)


class ContextBudgeter:
    """Greedily packs prioritized sections of rank-ordered items into a token
    budget. Never truncates mid-item — a section either keeps a whole item or
    drops it. Sections are granted their `min_tokens` first, then remaining
    budget is spent in priority order (lower number = higher priority) up to
    each section's `max_tokens`.
    """

    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self._sections: list[Section] = []

    def add(self, section: Section) -> None:
        self._sections.append(section)

    def pack(self) -> PackedPrompt:
        remaining = self.total_budget
        kept: dict[str, list[str]] = {}
        reports: list[SectionReport] = []

        # Pass 1: grant minimums (in declared order).
        for s in self._sections:
            items, tokens = self._fill(s.items, min(s.min_tokens, remaining if s.max_tokens is None else s.max_tokens))
            kept[s.name] = items
            remaining -= tokens

        # Pass 2: spend the rest in priority order, extending each section
        # up to its max_tokens (or the remaining budget if no max).
        for s in sorted(self._sections, key=lambda s: s.priority):
            already = kept[s.name]
            already_tokens = estimate_tokens(s.joiner.join(already))
            cap = s.max_tokens if s.max_tokens is not None else float("inf")
            budget_for_section = min(cap - already_tokens, remaining)
            if budget_for_section <= 0:
                continue
            extra_items, extra_tokens = self._fill(
                s.items[len(already):], budget_for_section
            )
            kept[s.name] = already + extra_items
            remaining -= extra_tokens

        parts = []
        for s in self._sections:
            items = kept[s.name]
            text = s.joiner.join(items)
            tokens = estimate_tokens(text)
            if s.items and not items:
                warnings.warn(
                    f"ContextBudgeter: section '{s.name}' dropped ALL {len(s.items)} item(s) "
                    f"-- its content doesn't fit within max_tokens={s.max_tokens}. "
                    "If this section carries required instructions (e.g. system rules), "
                    "raise its min_tokens/max_tokens to comfortably exceed the section's "
                    "actual measured token size -- sections are never partially truncated.",
                    stacklevel=2,
                )
            reports.append(SectionReport(
                name=s.name,
                items_total=len(s.items),
                items_kept=len(items),
                tokens=tokens,
                truncated=len(items) < len(s.items),
            ))
            if text:
                parts.append(text)

        return PackedPrompt(prompt="\n\n".join(parts), report=reports)

    @staticmethod
    def _fill(items: list[str], budget) -> tuple[list[str], int]:
        kept: list[str] = []
        used = 0
        for item in items:
            t = estimate_tokens(item)
            if used + t > budget:
                break
            kept.append(item)
            used += t
        return kept, used
