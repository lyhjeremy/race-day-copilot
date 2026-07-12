"""Overnight resumable distillation-data generator for Race Day Copilot.

Scenario grid (RACE_DAY_COPILOT_SPEC.md §6): goal x temp x course-archetype x
experience. Teacher = claude -p (smart tier) using the identical prompt
assembly as the live app. Only plans that pass verifier.py enter the training
set -- quality-filtered distillation. Rejects go to scenarios.rejected.jsonl
(teacher acceptance rate is a reported eval stat, not hidden).

Run under caffeinate so overnight sleep doesn't kill it:
  caffeinate -i python scripts/gen_scenarios.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gen_data import DatasetGenerator
from guardrails import GuardrailError, generate_validated
from schemas import CourseProfile, PacingPlan, RunnerBrief
from copilot import assemble_prompt
from verifier import verify

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "example_courses"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "scenarios.jsonl"

GOALS_MIN = list(range(165, 361, 15))  # 2:45 to 6:00, 15-min steps (14 values, marathon-scale)
HALF_GOALS_MIN = list(range(75, 181, 10))  # half-marathon-scale goals
TEMPS_C = [8, 15, 18, 22, 26, 30]
EXPERIENCE = ["first", "1-3", "4+"]


def _load_courses() -> dict[str, CourseProfile]:
    courses = {}
    for path in ASSETS_DIR.glob("*.json"):
        courses[path.stem] = CourseProfile.model_validate_json(path.read_text())
    return courses


def _stable_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def build_scenario_grid() -> list[dict]:
    courses = _load_courses()
    scenarios = []
    for course_name, course in courses.items():
        goals = HALF_GOALS_MIN if course.distance_km < 30 else GOALS_MIN
        for goal, temp, exp in itertools.product(goals, TEMPS_C, EXPERIENCE):
            item_id = _stable_id(course_name, goal, temp, exp)
            scenarios.append({
                "id": item_id,
                "course_name": course_name,
                "goal_time_min": goal,
                "temp_c": temp,
                "experience": exp,
            })
    return scenarios


def build_prompt(item: dict) -> str:
    # Stashed on the item by the caller (see main()) since DatasetGenerator's
    # build_prompt only receives the item dict, not external state.
    return item["_prompt"]


def parse(raw: str) -> dict:
    return json.loads(raw)


def validate(parsed: dict) -> list[str]:
    return parsed.get("_violations", [])


def main():
    courses = _load_courses()
    scenarios = build_scenario_grid()

    # Precompute prompts + retrieved ids per scenario, and stash a generate_fn
    # closure per item so each scenario runs the FULL guardrails.generate_validated
    # + verifier loop (not raw llm.generate) -- matching the live app exactly.
    for item in scenarios:
        course = courses[item["course_name"]]
        brief = RunnerBrief(goal_time_min=item["goal_time_min"], temp_c=item["temp_c"],
                             experience=item["experience"])
        prompt, packed, retrieved_ids = assemble_prompt(course, brief)
        item["_prompt"] = prompt
        item["_course_json"] = course.model_dump_json()
        item["_brief_json"] = brief.model_dump_json()
        item["_retrieved_ids"] = retrieved_ids

    def generate_fn(prompt: str) -> str:
        # find the item this prompt belongs to (small grid lookup by id done
        # via closure below in a wrapper -- see call site)
        raise NotImplementedError  # replaced per-item below

    # DatasetGenerator's generate_fn only takes `prompt`; we need per-item
    # course/brief/retrieved_ids for the verifier, so wrap generate_fn to look
    # them up from a side dict keyed by prompt text.
    by_prompt = {item["_prompt"]: item for item in scenarios}

    def generate_fn(prompt: str) -> str:
        item = by_prompt[prompt]
        course = CourseProfile.model_validate_json(item["_course_json"])
        brief = RunnerBrief.model_validate_json(item["_brief_json"])
        retrieved_ids = item["_retrieved_ids"]

        try:
            plan = generate_validated(
                prompt, PacingPlan,
                verifier=lambda p: verify(p, course, brief, retrieved_ids=retrieved_ids),
                max_retries=2,
                # Measured live (2026-07-11 smoke test): a full 43-row plan takes
                # ~257s at the "smart" tier -- well above the toolkit's 120s
                # default. timeout_s/max_tokens raised accordingly.
                llm_kwargs={"tier": "smart", "timeout_s": 280, "max_tokens": 3000},
            )
        except GuardrailError as e:
            # teacher couldn't produce a passing plan -- reject, don't fail
            # (failures are reserved for actual API/network errors, which
            # signal rate-limiting and should stop the run).
            return json.dumps({
                "course_name": item["course_name"], "goal_time_min": item["goal_time_min"],
                "temp_c": item["temp_c"], "experience": item["experience"],
                "_violations": e.violations or ["exhausted retries"],
            })

        return json.dumps({
            "course_name": item["course_name"], "goal_time_min": item["goal_time_min"],
            "temp_c": item["temp_c"], "experience": item["experience"],
            "plan": plan.model_dump(), "_violations": [],
        })

    gen = DatasetGenerator(
        name="copilot_scenarios", out_path=OUT_PATH, items=scenarios,
        build_prompt=build_prompt, parse=parse, validate=validate, generate_fn=generate_fn,
    )
    gen.run(max_consecutive_failures=3, sleep_between=1.5)


if __name__ == "__main__":
    main()
