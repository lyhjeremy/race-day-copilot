"""Pacing-plan scenario -> mlx-lm chat-format train/valid/test splits. See
RACE_DAY_COPILOT_SPEC.md §7: split by scenario id (item_id), ~90/5/5, held-out
~10% becomes the 3-way benchmark set. Reconstructs the EXACT teacher prompt via
copilot.assemble_prompt so student and teacher see identical input at bench time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training" / "lora_harness"))

from prep import prep_dataset  # noqa: E402
from copilot import assemble_prompt  # noqa: E402
from schemas import CourseProfile, RunnerBrief  # noqa: E402

SCENARIOS_PATH = ROOT / "data" / "scenarios.jsonl"
COURSES_DIR = ROOT / "assets" / "example_courses"
OUT_DIR = ROOT / "data" / "lora"
EVAL_DIR = ROOT / "eval"
# RACE_DAY_COPILOT_SPEC.md §7 / TOOLKIT_SPEC.md §8.2 suggest max-seq=1536 for the
# Copilot distill, but that's sized for a much shorter completion than this task
# actually produces. Measured with the real Qwen2.5 tokenizer (2026-07-22): the
# worst-case example (prompt + full ~43-row marathon plan JSON) is 4,197 tokens,
# 2.7x over that budget -- would silently truncate mid-example if trained as-is.
# Raised to 4608 (real max + headroom) rather than trimming the prompt/output.
MAX_SEQ_LEN = 4608


def _load_courses() -> dict[str, CourseProfile]:
    return {p.stem: CourseProfile.model_validate_json(p.read_text())
            for p in COURSES_DIR.glob("*.json")}


def load_scenarios() -> list[dict]:
    if not SCENARIOS_PATH.exists():
        raise SystemExit(f"{SCENARIOS_PATH} not found -- run scripts/gen_scenarios.py first.")
    rows = [json.loads(l) for l in SCENARIOS_PATH.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("_violations")]
    return rows


def build_dataset() -> dict:
    courses = _load_courses()
    rows = load_scenarios()
    print(f"{len(rows)} accepted scenarios loaded")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    # Reconstruct the (prompt, plan) pair for every row once, up front, so we
    # can both build the mlx-lm messages AND assert max-seq-length fit (with the
    # REAL tokenizer, not a chars/token estimate) before burning GPU time on a
    # training run that would truncate mid-example.
    enriched = []
    max_tokens = 0
    for r in rows:
        course = courses[r["course_name"]]
        brief = RunnerBrief(goal_time_min=r["goal_time_min"], temp_c=r["temp_c"],
                             experience=r["experience"])
        prompt, _packed, _retrieved_ids = assemble_prompt(course, brief)
        assistant = json.dumps(r["plan"], ensure_ascii=False)
        enriched.append({"item_id": r["item_id"], "_prompt": prompt, "_assistant": assistant})
        n = len(tok.encode(prompt)) + len(tok.encode(assistant))
        max_tokens = max(max_tokens, n)

    print(f"Longest example: {max_tokens} tokens (real Qwen2.5 tokenizer), budget {MAX_SEQ_LEN}")
    assert max_tokens < MAX_SEQ_LEN * 0.95, (
        f"Longest training example ({max_tokens} tokens) is too close to "
        f"max-seq-length={MAX_SEQ_LEN} -- would truncate mid-example. Raise seq_len "
        "or trim the prompt before training."
    )

    def to_messages(rec: dict) -> list[dict]:
        return [
            {"role": "user", "content": rec["_prompt"]},
            {"role": "assistant", "content": rec["_assistant"]},
        ]

    card = prep_dataset(
        enriched,
        entity_key_fn=lambda r: r["item_id"],  # split by scenario id per spec §7
        to_messages_fn=to_messages,
        out_dir=OUT_DIR,
    )
    card["max_example_tokens"] = max_tokens
    card["max_seq_length_budget"] = MAX_SEQ_LEN

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return card


if __name__ == "__main__":
    print(json.dumps(build_dataset(), indent=2))
