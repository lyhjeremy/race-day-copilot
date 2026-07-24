"""3-way benchmark for Race Day Copilot: base Qwen vs LoRA-fine-tuned Qwen vs
Claude teacher, on the held-out test split. Per RACE_DAY_COPILOT_SPEC.md §7:

1. Objective (zero-shot, no retry): verifier pass rate + per-check rates for
   all three systems, plus parse-success rate (a system that can't even
   produce valid JSON fails everything downstream -- reported explicitly,
   not hidden).
2. LLM-judge: blind pairwise (position-swapped) of LoRA vs teacher on a
   sampled subset -- rubric: realism, clarity of notes, safety of advice.
3. Latency/cost: s/plan for local (measured) vs teacher; $/1k plans.
4. Per-check failure analysis: which constraints the student still breaks.

Held-out set is recomputed independently from data/scenarios.jsonl using the
exact same entity-bucket split as training/lora_harness/prep.py (seed=42,
90/5/5) -- NOT parsed from data/lora/test.jsonl, since that's already in
mlx-lm chat-message format and doesn't carry the structured course/brief/plan
objects the verifier needs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training" / "lora_harness"))

import llm  # noqa: E402
from copilot import assemble_prompt  # noqa: E402
from guardrails import _tolerant_json_parse  # noqa: E402
from schemas import CourseProfile, PacingPlan, RunnerBrief  # noqa: E402
from verifier import verify  # noqa: E402

COURSES_DIR = ROOT / "assets" / "example_courses"
SCENARIOS_PATH = ROOT / "data" / "scenarios.jsonl"
EVAL_DIR = ROOT / "eval"
BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
LORA_ADAPTER = ROOT / "training" / "adapters_v2" / "adapters.safetensors"
SEED = 42
JUDGE_N = 60


def _entity_bucket(entity_key: str, n_buckets: int = 1000) -> int:
    h = hashlib.sha256(entity_key.encode("utf-8")).hexdigest()
    return int(h, 16) % n_buckets


def load_test_scenarios() -> list[dict]:
    courses = {p.stem: CourseProfile.model_validate_json(p.read_text())
               for p in COURSES_DIR.glob("*.json")}
    rows = [json.loads(l) for l in SCENARIOS_PATH.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("_violations")]

    test_rows = []
    for r in rows:
        bucket = _entity_bucket(f"{SEED}:{r['item_id']}")
        if bucket >= 950:  # matches prep.py's default 90/5/5 split_ratios
            test_rows.append(r)

    enriched = []
    for r in test_rows:
        course = courses[r["course_name"]]
        brief = RunnerBrief(goal_time_min=r["goal_time_min"], temp_c=r["temp_c"],
                             experience=r["experience"])
        prompt, _packed, retrieved_ids = assemble_prompt(course, brief)
        enriched.append({
            "item_id": r["item_id"], "course_name": r["course_name"],
            "course": course, "brief": brief, "prompt": prompt,
            "retrieved_ids": retrieved_ids, "reference_plan": r["plan"],
        })
    return enriched


def parse_plan(raw: str) -> tuple[PacingPlan | None, str | None]:
    try:
        data = _tolerant_json_parse(raw)
        return PacingPlan.model_validate(data), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run_mlx(prompt: str, *, adapter_path: str | None = None, max_tokens: int = 3000) -> tuple[str, float]:
    args = ["mlx_lm.generate", "--model", BASE_MODEL, "--prompt", prompt,
            "--max-tokens", str(max_tokens)]
    if adapter_path:
        args += ["--adapter-path", adapter_path]
    start = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, timeout=300)
    latency = time.time() - start
    out = proc.stdout
    # mlx_lm.generate CLI wraps the actual generation between two "======"
    # separator lines, followed by a stats banner (Prompt:/Generation:/Peak
    # memory:) after the second separator -- the generated text is the
    # middle section, NOT "everything after the last separator" (that's the
    # stats banner and was a real bug caught by a smoke test: it returned
    # the stats banner alone, with 0% JSON parse success, on every item).
    parts = out.split("==========")
    if len(parts) >= 3:
        out = parts[1]
    return out.strip(), latency


def run_teacher(prompt: str) -> tuple[str, float]:
    resp = llm.generate(prompt, tier="smart", timeout_s=280, max_tokens=3000)
    return resp.text, resp.latency_s


VIOLATION_CATEGORIES = [
    ("parse", None),  # handled separately
    ("coverage", "coverage is wrong"),
    ("sum", "Split total"),
    ("plausibility", "deviates"),
    ("plausibility", "implausibly fast"),
    ("fade", "Fade is"),
    ("fade", "fade_allowance_pct"),
    ("fade", "negative split"),
    ("heat", "heat_adjustment_s_per_km"),
    ("citations", "citations"),
]


def categorize(violations: list[str]) -> set[str]:
    cats = set()
    for v in violations:
        for cat, needle in VIOLATION_CATEGORIES:
            if needle and needle in v:
                cats.add(cat)
    return cats


def eval_system(name: str, items: list[dict], gen_fn) -> dict:
    n = len(items)
    parse_ok = 0
    verify_pass = 0
    violation_cat_counts = Counter()
    latencies = []
    raw_outputs = []

    for i, item in enumerate(items):
        print(f"[{name}] {i+1}/{n} ({item['item_id']})", flush=True)
        raw, latency = gen_fn(item["prompt"])
        latencies.append(latency)
        plan, err = parse_plan(raw)
        # Full plans run 4-5k+ chars; a 4000-char truncation here (an earlier
        # version of this script) silently corrupted the SAVED raw_outputs
        # JSON mid-string for most items, even though scoring above already
        # ran on the untruncated `raw` -- the scored metrics were correct,
        # but the saved files were unusable for exactly the "read raw output
        # directly" audit this project's own reliability lessons call for.
        # 20000 comfortably covers this task's outputs.
        raw_outputs.append({"item_id": item["item_id"], "raw": raw[:20000], "parse_error": err})
        if plan is None:
            violation_cat_counts["parse"] += 1
            continue
        parse_ok += 1
        violations = verify(plan, item["course"], item["brief"], retrieved_ids=item["retrieved_ids"])
        if not violations:
            verify_pass += 1
        else:
            for cat in categorize(violations):
                violation_cat_counts[cat] += 1

    (EVAL_DIR / f"raw_outputs_{name}.json").write_text(json.dumps(raw_outputs, indent=2, ensure_ascii=False))

    return {
        "system": name,
        "n": n,
        "parse_success_rate": round(parse_ok / n, 3),
        "verifier_pass_rate": round(verify_pass / n, 3),
        "violation_rates": {cat: round(cnt / n, 3) for cat, cnt in violation_cat_counts.items()},
        "mean_latency_s": round(sum(latencies) / n, 2),
    }


def blind_pairwise_judge(item_a: str, item_b: str, rubric: str) -> str:
    prompt = (
        f"{rubric}\n\nOption A:\n{item_a}\n\nOption B:\n{item_b}\n\n"
        "Which is better? Respond with exactly one word: 'A', 'B', or 'tie'."
    )
    resp = llm.generate(prompt, tier="smart", max_tokens=10)
    verdict = resp.text.strip().lower()
    if verdict.startswith("a"):
        return "a"
    if verdict.startswith("b"):
        return "b"
    return "tie"


def judge_pair_swapped(student_output: str, teacher_output: str, rubric: str) -> str:
    v1 = blind_pairwise_judge(student_output, teacher_output, rubric)
    v2 = blind_pairwise_judge(teacher_output, student_output, rubric)
    v1_pick = {"a": "student", "b": "teacher", "tie": "tie"}[v1]
    v2_pick = {"a": "teacher", "b": "student", "tie": "tie"}[v2]
    if v1_pick == v2_pick:
        return v1_pick
    return "tie"


def run_judge(items: list[dict], lora_outputs: dict[str, str], teacher_outputs: dict[str, str]) -> dict:
    rubric = (
        "You are judging two marathon/half-marathon pacing plans for the same "
        "runner and course. Judge on: realism of the pacing (does it make sense "
        "given the course profile and goal), clarity of the per-km notes, and "
        "safety of any cautions/advice given."
    )
    n = min(JUDGE_N, len(items))
    sample = items[:n]
    results = []
    counts = Counter()
    for i, item in enumerate(sample):
        iid = item["item_id"]
        lora_out = lora_outputs.get(iid)
        teacher_out = teacher_outputs.get(iid)
        if not lora_out or not teacher_out:
            continue
        print(f"[judge] {i+1}/{n} ({iid})", flush=True)
        verdict = judge_pair_swapped(lora_out, teacher_out, rubric)
        counts[verdict] += 1
        results.append({"item_id": iid, "verdict": verdict})
    (EVAL_DIR / "judge_outputs.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    total = sum(counts.values())
    return {
        "n_judged": total,
        "student_wins": counts.get("student", 0),
        "teacher_wins": counts.get("teacher", 0),
        "ties": counts.get("tie", 0),
        "student_win_rate": round(counts.get("student", 0) / total, 3) if total else None,
        "teacher_win_rate": round(counts.get("teacher", 0) / total, 3) if total else None,
        "tie_rate": round(counts.get("tie", 0) / total, 3) if total else None,
    }


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    items = load_test_scenarios()
    print(f"Held-out test set: {len(items)} scenarios")

    lora_raw: dict[str, str] = {}
    teacher_raw: dict[str, str] = {}

    def base_gen(prompt):
        return run_mlx(prompt, adapter_path=None)

    def lora_gen(prompt):
        raw, lat = run_mlx(prompt, adapter_path=str(LORA_ADAPTER.parent))
        return raw, lat

    def teacher_gen(prompt):
        raw, lat = run_teacher(prompt)
        return raw, lat

    base_report = eval_system("base", items, base_gen)
    print(json.dumps(base_report, indent=2))

    lora_report = eval_system("lora", items, lora_gen)
    print(json.dumps(lora_report, indent=2))
    for item, rec in zip(items, json.loads((EVAL_DIR / "raw_outputs_lora.json").read_text())):
        lora_raw[rec["item_id"]] = rec["raw"]

    teacher_report = eval_system("teacher", items, teacher_gen)
    print(json.dumps(teacher_report, indent=2))
    for rec in json.loads((EVAL_DIR / "raw_outputs_teacher.json").read_text()):
        teacher_raw[rec["item_id"]] = rec["raw"]

    judge_report = run_judge(items, lora_raw, teacher_raw)
    print(json.dumps(judge_report, indent=2))

    cost_per_1k = {
        "base": 0.0, "lora": 0.0,
        "teacher": None,  # claude -p via Max subscription, not per-token billed
    }

    report = {
        "held_out_n": len(items),
        "systems": {"base": base_report, "lora": lora_report, "teacher": teacher_report},
        "judge": judge_report,
        "cost_per_1k_plans_usd": cost_per_1k,
    }
    (EVAL_DIR / "benchmark_raw.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = ["# Race Day Copilot — 3-way benchmark", "",
             f"Held-out test set: **{len(items)} scenarios** (split by scenario id, seed=42, 90/5/5).", "",
             "## Objective: parse success & verifier pass rate (zero-shot, no retry)", "",
             "| System | Parse success | Verifier pass rate | Mean latency (s/plan) |",
             "|---|---|---|---|"]
    for sysname in ["base", "lora", "teacher"]:
        r = report["systems"][sysname]
        lines.append(f"| {sysname} | {r['parse_success_rate']:.1%} | {r['verifier_pass_rate']:.1%} | {r['mean_latency_s']:.2f} |")
    lines += ["", "## Per-check violation rates (share of items failing each check)", "",
              "| System | " + " | ".join(sorted(set(
                  cat for r in report["systems"].values() for cat in r["violation_rates"]))) + " |"]
    all_cats = sorted(set(cat for r in report["systems"].values() for cat in r["violation_rates"]))
    lines.append("|---|" + "---|" * len(all_cats))
    for sysname in ["base", "lora", "teacher"]:
        r = report["systems"][sysname]
        row = [f"{r['violation_rates'].get(c, 0):.1%}" for c in all_cats]
        lines.append(f"| {sysname} | " + " | ".join(row) + " |")
    lines += ["", "## LLM judge: LoRA vs teacher (blind, position-swapped)", "",
              f"N judged: {judge_report['n_judged']}", "",
              f"- Student (LoRA) preferred: {judge_report['student_win_rate']:.1%}" if judge_report['student_win_rate'] is not None else "- (no judged pairs)",
              f"- Teacher preferred: {judge_report['teacher_win_rate']:.1%}" if judge_report['teacher_win_rate'] is not None else "",
              f"- Tie: {judge_report['tie_rate']:.1%}" if judge_report['tie_rate'] is not None else ""]
    (EVAL_DIR / "benchmark.md").write_text("\n".join(lines))
    print("\nWrote eval/benchmark.md, eval/benchmark_raw.json, eval/judge_outputs.json, eval/raw_outputs_*.json")


if __name__ == "__main__":
    main()
