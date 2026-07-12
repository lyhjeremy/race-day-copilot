"""Prompt assembly for the pacing-plan generation task. Shared by the live
app and the scenario generator/benchmark so teacher and student always see
the identical prompt shape (RACE_DAY_COPILOT_SPEC.md §4).
"""
from __future__ import annotations

import json

from context import ContextBudgeter, Section
from retriever import retrieve
from schemas import CourseProfile, RunnerBrief

SYSTEM_RULES = """You are Race Day Copilot, a marathon/half-marathon pacing coach.
You will be given a course profile, a runner's goal, and retrieved knowledge cards.
Produce a km-by-km pacing plan as JSON matching the PacingPlan schema.

Rules you MUST follow:
- Every km from 1 to floor(distance) gets exactly one split row, PLUS one final
  partial row for the remainder (e.g. km=43 for a 42.2km course covering the
  final 0.2km) if the course isn't a whole number of km.
- target_pace is ALWAYS a per-kilometer PACE RATE in mm:ss format (e.g. "5:41"
  means 5 minutes 41 seconds per kilometer) -- this applies to EVERY row,
  INCLUDING the final partial row. Do NOT give the partial row's actual elapsed
  time for its shorter distance; give its pace rate, exactly like every other
  row (e.g. if you're running 5:41/km at that point in the race, the partial
  row's target_pace is still "5:41", not a scaled-down time). predicted_finish
  and cumulative use mm:ss or h:mm:ss format for actual elapsed time (not a rate).
- Adjust pace for hills: hold effort not pace (slower uphill, faster downhill,
  within reason) -- see the segment list in the course profile.
- Heat adjustment: if temp_c >= 18, add heat_adjustment_s_per_km using this
  formula band: between 0.75*(temp_c-15) and 4.5*(temp_c-15) seconds per km,
  derived from Jeremy Lee's Boston Marathon heat-tax finding (~1 min slower per
  degree F on the field median, r=0.86). Apply this adjustment by slowing every
  split proportionally. If temp_c < 18, use 0-3 s/km.
- Fade: plan for 0-8% slower second half than first half (the honest median
  outcome per the pacing-decay knowledge card), UNLESS the runner's goal and
  experience genuinely support an even or negative-split attempt -- in that
  case explicitly say so in strategy_summary AND add a caution citing that only
  2.5% of runners achieve a true negative split.
- If the goal appears more than 2 standard deviations faster than what
  recent_race implies (Riegel-style), don't blindly comply: say so plainly in
  strategy_summary, generate the plan for a more realistic target instead, and
  make predicted_finish match your actual split total (not the stated goal).
- citations: list the ids of the knowledge cards you actually used. Never
  invent a citation id that wasn't given to you.
- Never give specific medical advice; if asked about injury/pain, add a
  caution recommending they consult a medical professional instead.
"""


def build_retrieval_query(course: CourseProfile, brief: RunnerBrief) -> str:
    seg_desc = ", ".join(f"{s.trend}/{s.severity} km{s.km_start}-{s.km_end}" for s in course.segments)
    parts = [f"{course.distance_km}km course, segments: {seg_desc}",
             f"goal {brief.goal_time_min}min"]
    if brief.temp_c is not None:
        parts.append(f"{brief.temp_c}C")
    if brief.experience:
        parts.append(f"experience {brief.experience}")
    return ", ".join(parts)


def assemble_prompt(course: CourseProfile, brief: RunnerBrief, *, total_budget: int = 3200):
    """Returns (prompt_text, packed_report, retrieved_ids) -- retrieved_ids
    feeds verifier.verify()'s citation-subset check.

    NOTE: SYSTEM_RULES is fixed content (measured at 661 tokens as of the
    target_pace-semantics fix) containing safety-critical instructions (heat
    formula, fade rules, pace-rate convention, citation rules). Its
    min/max_tokens below are set with real headroom above the measured size
    -- a section's max_tokens MUST exceed its actual content size or
    ContextBudgeter drops the item entirely (no mid-item truncation). This
    exact bug bit this file twice during Session 1 testing (once at 529
    tokens vs max=500, again at 661 vs max=650 after editing the rules text)
    -- the toolkit's own drop-warning caught both. If you edit SYSTEM_RULES
    again, re-run this module's smoke test (see RACE_DAY_COPILOT_SPEC.md) and
    watch for the UserWarning before trusting the prompt.
    """
    query = build_retrieval_query(course, brief)
    retrieved = retrieve(query, k=6)
    retrieved_ids = [c["id"] for c in retrieved]

    budgeter = ContextBudgeter(total_budget=total_budget)
    budgeter.add(Section(name="system", items=[SYSTEM_RULES], priority=0, min_tokens=800, max_tokens=800))
    budgeter.add(Section(
        name="course_and_brief",
        items=[f"Course profile:\n{course.model_dump_json(indent=2)}\n\nRunner brief:\n{brief.model_dump_json(indent=2)}"],
        priority=0, min_tokens=350, max_tokens=400,
    ))
    budgeter.add(Section(
        name="knowledge_cards",
        items=[f"[{c['id']}] {c['text']}" for c in retrieved],
        priority=2, max_tokens=1200,
    ))
    budgeter.add(Section(
        name="output_format",
        items=[
            "Respond with ONLY a JSON object matching the PacingPlan schema: "
            "strategy_summary, splits (list of {km, target_pace, cumulative, note}), "
            "predicted_finish, fade_allowance_pct, heat_adjustment_s_per_km, "
            "cautions (list of strings), citations (list of knowledge card ids used)."
        ],
        priority=1, min_tokens=100,
    ))

    packed = budgeter.pack()
    return packed.prompt, packed, retrieved_ids
