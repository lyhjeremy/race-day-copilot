"""Deterministic arithmetic verifier for PacingPlan. See RACE_DAY_COPILOT_SPEC.md §2.

Plugs into guardrails.generate_validated(schema=PacingPlan, verifier=verify).
Every violation is a complete, human-readable sentence -- these get injected
verbatim into the regeneration prompt, so they must be self-explanatory to
an LLM with no other context.
"""
from __future__ import annotations

import math
import re

from schemas import CourseProfile, PacingPlan, RunnerBrief

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")


def parse_time_to_seconds(s: str) -> int | None:
    m = _TIME_RE.match(s.strip())
    if not m:
        return None
    h, mm, ss = m.groups()
    h = int(h) if h else 0
    return h * 3600 + int(mm) * 60 + int(ss)


def _segment_for_km(course: CourseProfile, km: float):
    for seg in course.segments:
        if seg.km_start <= km < seg.km_end or (km == course.distance_km and seg.km_end == km):
            return seg
    return None


def verify(plan: PacingPlan, course: CourseProfile, brief: RunnerBrief,
           retrieved_ids: list[str] | None = None) -> list[str]:
    violations: list[str] = []

    # --- parse all split times up front; hard-fail early if unparseable ---
    split_secs: dict[int, int] = {}
    for row in plan.splits:
        secs = parse_time_to_seconds(row.target_pace)
        if secs is None:
            violations.append(
                f"Split at km={row.km} has an unparseable target_pace '{row.target_pace}' "
                "(expected mm:ss or h:mm:ss)."
            )
            continue
        split_secs[row.km] = secs

    if violations:
        return violations  # can't run further checks on unparseable data

    # --- 1. Coverage ---
    full_kms = list(range(1, math.floor(course.distance_km) + 1))
    remainder = course.distance_km - math.floor(course.distance_km)
    expected_kms = full_kms + ([math.floor(course.distance_km) + 1] if remainder > 0.05 else [])
    got_kms = sorted(split_secs.keys())
    if got_kms != expected_kms:
        missing = set(expected_kms) - set(got_kms)
        extra = set(got_kms) - set(expected_kms)
        parts = []
        if missing:
            parts.append(f"missing km rows {sorted(missing)}")
        if extra:
            parts.append(f"unexpected/duplicate km rows {sorted(extra)}")
        violations.append(
            f"Split coverage is wrong for a {course.distance_km}km course: {', '.join(parts)}. "
            f"Expected exactly one row per km: {expected_kms}."
        )
        return violations  # further checks assume complete coverage

    # --- 2. Sum check ---
    # target_pace is ALWAYS a per-km rate (e.g. "5:41" = 5min41s per km),
    # for every row including the partial final one -- NOT the actual
    # elapsed time for that row's distance. This matches how real running
    # apps report splits and is stated explicitly in SYSTEM_RULES. Elapsed
    # time therefore = rate * distance-of-that-row (1km for full rows,
    # `remainder` for the partial row). A live-generation smoke test caught
    # this: without this correction, the model's partial-row pace value
    # (a normal ~4-6min/km rate) got compared directly against full-km
    # elapsed times, or scaled by 1/remainder into a nonsensical ~20min/km
    # "deviation", rejecting valid plans.
    partial_km = expected_kms[-1] if remainder > 0.05 else None
    elapsed_secs = {
        km: secs * (remainder if km == partial_km else 1)
        for km, secs in split_secs.items()
    }
    total_secs = sum(elapsed_secs.values())
    goal_secs = brief.goal_time_min * 60
    goal_flagged_unrealistic = "aggressive" in plan.strategy_summary.lower() or any(
        "unrealistic" in c.lower() or "aggressive" in c.lower() for c in plan.cautions
    )
    if goal_flagged_unrealistic:
        predicted_secs = parse_time_to_seconds(plan.predicted_finish)
        if predicted_secs is not None and abs(total_secs - predicted_secs) > 90:
            violations.append(
                f"Split total ({total_secs}s) does not match the plan's own predicted_finish "
                f"({plan.predicted_finish} = {predicted_secs}s) within 90s tolerance."
            )
    elif abs(total_secs - goal_secs) > 90:
        violations.append(
            f"Split total is {total_secs}s but the goal is {goal_secs}s "
            f"({brief.goal_time_min} min) -- off by {abs(total_secs - goal_secs)}s, "
            "more than the 90s tolerance. Either fix the splits to sum to the goal, "
            "or explicitly flag the goal as aggressive/unrealistic in strategy_summary "
            "and cautions, and make predicted_finish match the split total instead."
        )

    # --- 3. Split plausibility vs segment ---
    # split_secs is already a per-km rate for every row (including the
    # partial one -- see §2 note), so no normalization is needed here.
    mean_pace = sum(split_secs.values()) / len(split_secs)
    for km, secs in split_secs.items():
        seg = _segment_for_km(course, km)
        deviation = (secs - mean_pace) / mean_pace
        if seg and seg.trend == "up" and seg.severity == "steep":
            allowed = 0.20
        elif seg and seg.trend == "down":
            allowed = -0.08  # splits should be faster (negative deviation), tolerate down to -8%
            if deviation < allowed - 0.12:  # still cap extreme downhill speedup
                violations.append(
                    f"Split at km={km} ({secs}s) is implausibly fast for a downhill segment "
                    f"(deviation {deviation:.1%} vs course mean pace)."
                )
            continue
        else:
            allowed = 0.12
        if deviation > allowed:
            violations.append(
                f"Split at km={km} ({secs}s) deviates {deviation:.1%} from the mean pace "
                f"({mean_pace:.0f}s/km), exceeding the {allowed:.0%} allowance for this segment "
                f"(trend={seg.trend if seg else 'unknown'}, severity={seg.severity if seg else 'n/a'})."
            )

    # --- 4. Fade direction (split_secs are already per-km rates, see §2) ---
    half_km = course.distance_km / 2
    first_half_secs = sum(s for km, s in split_secs.items() if km <= half_km)
    second_half_secs = sum(s for km, s in split_secs.items() if km > half_km)
    first_half_km = sum(1 for km in split_secs if km <= half_km)
    second_half_km = len(split_secs) - first_half_km
    if first_half_km and second_half_km:
        first_pace = first_half_secs / first_half_km
        second_pace = second_half_secs / second_half_km
        fade_pct = (second_pace - first_pace) / first_pace * 100

        negative_split_flagged = any(
            "negative split" in c.lower() and ("2.5%" in c or "rare" in c.lower())
            for c in plan.cautions
        )
        if fade_pct < -0.5 and not negative_split_flagged:
            violations.append(
                f"Plan shows a negative split (second half {fade_pct:.1f}% vs first half) "
                "without flagging it as an aggressive attempt and citing the 2.5% base rate "
                "in a caution. Either add fade (0-8%) or add that caution explicitly."
            )
        elif fade_pct > 8.5:
            violations.append(
                f"Fade is {fade_pct:.1f}%, exceeding the 8% maximum allowance for planned fade."
            )
        if abs(fade_pct - plan.fade_allowance_pct) > 0.5:
            violations.append(
                f"fade_allowance_pct ({plan.fade_allowance_pct}) does not match the actual "
                f"split math ({fade_pct:.1f}%) within 0.5 percentage points."
            )

    # --- 5. Heat rule ---
    temp = brief.temp_c
    if temp is not None:
        if temp >= 18:
            lo, hi = 0.75 * (temp - 15), 4.5 * (temp - 15)
            if not (lo <= plan.heat_adjustment_s_per_km <= hi):
                violations.append(
                    f"heat_adjustment_s_per_km ({plan.heat_adjustment_s_per_km}) is outside the "
                    f"expected band [{lo:.1f}, {hi:.1f}] s/km for {temp}°C "
                    "(derived from ~1 min/°F on a median finisher, per repo 14's heat-tax finding)."
                )
        elif plan.heat_adjustment_s_per_km > 3 or plan.heat_adjustment_s_per_km < 0:
            violations.append(
                f"heat_adjustment_s_per_km ({plan.heat_adjustment_s_per_km}) should be 0-3 s/km "
                f"when temperature ({temp}°C) is below 18°C."
            )

    # --- 6. Citations ---
    if not plan.citations:
        violations.append(
            "citations list is empty -- every plan must cite at least one retrieved knowledge card."
        )
    elif retrieved_ids is not None:
        invented = [c for c in plan.citations if c not in retrieved_ids]
        if invented:
            violations.append(
                f"citations include ids not present in the retrieved set: {invented}. "
                f"Only cite from: {retrieved_ids}."
            )

    return violations
