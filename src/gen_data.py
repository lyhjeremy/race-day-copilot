"""Resumable overnight dataset generator. Generalizes SkillCompass's
gen_questions.py pattern: crash-safe append, idempotent resume by stable item
id, self-stop after N consecutive failures (usually = rate limit; rerun the
same command the next day).

Usage: run under `caffeinate -i` so sleep doesn't kill it, e.g.
  caffeinate -i python scripts/gen_scenarios.py
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class GenStats:
    done: int = 0
    skipped_existing: int = 0
    rejected: int = 0
    failed: int = 0


class DatasetGenerator:
    def __init__(self, name: str, out_path: str | Path,
                 items: list[dict],
                 build_prompt: Callable[[dict], str],
                 parse: Callable[[str], dict],
                 validate: Callable[[dict], list[str]] | None = None,
                 generate_fn: Callable[[str], str] | None = None):
        """
        items: each dict MUST carry a stable "id" key (hash of inputs, no
        timestamps/randomness) so resume can skip completed items.
        generate_fn: defaults to llm.generate(prompt, tier="smart").text if
        not supplied.
        """
        self.name = name
        self.out_path = Path(out_path)
        self.rejected_path = self.out_path.with_suffix(f".rejected.jsonl")
        self.items = items
        self.build_prompt = build_prompt
        self.parse = parse
        self.validate = validate
        self.generate_fn = generate_fn or self._default_generate

    @staticmethod
    def _default_generate(prompt: str) -> str:
        import llm
        return llm.generate(prompt, tier="smart", json_only=True).text

    def _existing_ids(self) -> set[str]:
        if not self.out_path.exists():
            return set()
        ids = set()
        for line in self.out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return ids

    def run(self, max_consecutive_failures: int = 3, sleep_between: float = 2.0) -> GenStats:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._existing_ids()
        stats = GenStats(skipped_existing=len(existing))
        consecutive_failures = 0

        with open(self.out_path, "a", encoding="utf-8") as out_f, \
             open(self.rejected_path, "a", encoding="utf-8") as rej_f:
            for item in self.items:
                item_id = item["id"]
                if item_id in existing:
                    continue

                try:
                    prompt = self.build_prompt(item)
                    raw = self.generate_fn(prompt)
                    parsed = self.parse(raw)
                except Exception as e:
                    stats.failed += 1
                    consecutive_failures += 1
                    print(f"[{self.name}] FAIL item {item_id}: {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        print(
                            f"[{self.name}] {consecutive_failures} consecutive failures "
                            f"(likely rate-limited). Stopping cleanly. "
                            f"Done: {stats.done}, remaining: "
                            f"{len(self.items) - stats.done - stats.skipped_existing - stats.rejected}. "
                            f"Rerun this command tomorrow to resume."
                        )
                        break
                    time.sleep(sleep_between * 4)
                    continue

                violations = self.validate(parsed) if self.validate else []
                if violations:
                    stats.rejected += 1
                    rej_f.write(json.dumps({"item_id": item_id, "violations": violations, "data": parsed}) + "\n")
                    rej_f.flush()
                    consecutive_failures = 0
                    continue

                record = {"item_id": item_id, **parsed}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                stats.done += 1
                consecutive_failures = 0

                if stats.done % 25 == 0:
                    print(f"[{self.name}] {stats.done} done, {stats.rejected} rejected, {stats.failed} failed")
                time.sleep(sleep_between)

        acceptance = stats.done / (stats.done + stats.rejected) if (stats.done + stats.rejected) else 0.0
        print(
            f"[{self.name}] run complete: {stats.done} written, {stats.rejected} rejected "
            f"(acceptance {acceptance:.1%}), {stats.failed} failed, {stats.skipped_existing} already done"
        )
        return stats
