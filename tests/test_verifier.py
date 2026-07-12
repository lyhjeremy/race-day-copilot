import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import CourseProfile, PacingPlan, RunnerBrief, Segment, SplitRow
from verifier import parse_time_to_seconds, verify

FLAT_COURSE = CourseProfile(
    distance_km=42.2,
    segments=[Segment(km_start=0, km_end=42.2, trend="flat", severity="gentle")],
    source_confidence=0.9,
)


def _flat_splits(pace_sec_per_km: int, n_full=42):
    # target_pace is a per-km RATE for every row, including the partial final
    # one (the corrected semantics -- see verifier.py's §2 comment). All rows,
    # partial included, get the same rate string for an "even pace" fixture.
    pace_str = f"{pace_sec_per_km // 60}:{pace_sec_per_km % 60:02d}"
    rows = [SplitRow(km=i, target_pace=pace_str, cumulative="0:00:00") for i in range(1, n_full + 1)]
    rows.append(SplitRow(km=n_full + 1, target_pace=pace_str, cumulative="0:00:00"))
    return rows


def test_parse_time_to_seconds():
    assert parse_time_to_seconds("5:41") == 341
    assert parse_time_to_seconds("3:45:00") == 13500
    assert parse_time_to_seconds("garbage") is None


def test_valid_plan_passes():
    # goal 240 min = 14400s; 42.2km even pace ~= 341.3s/km
    pace = 341
    total = pace * 42 + round(pace * 0.2)
    plan = PacingPlan(
        strategy_summary="Even pacing throughout, no aggressive negative split attempted.",
        splits=_flat_splits(pace),
        predicted_finish="4:00:00",
        fade_allowance_pct=0.0,
        heat_adjustment_s_per_km=0.0,
        citations=["even-pacing-strategy"],
    )
    brief = RunnerBrief(goal_time_min=round(total / 60), temp_c=12)
    violations = verify(plan, FLAT_COURSE, brief, retrieved_ids=["even-pacing-strategy", "heat-tax"])
    assert violations == []


def test_sum_mismatch_flagged():
    plan = PacingPlan(
        strategy_summary="Even pacing.",
        splits=_flat_splits(341),
        predicted_finish="2:00:00",  # wildly wrong, not flagged as aggressive
        fade_allowance_pct=0.0,
        heat_adjustment_s_per_km=0.0,
        citations=["even-pacing-strategy"],
    )
    brief = RunnerBrief(goal_time_min=350, temp_c=12)  # goal way off from splits
    violations = verify(plan, FLAT_COURSE, brief)
    assert any("Split total" in v for v in violations)


def test_missing_citations_flagged():
    pace = 341
    plan = PacingPlan(
        strategy_summary="Even pacing.",
        splits=_flat_splits(pace),
        predicted_finish="4:00:00",
        fade_allowance_pct=0.0,
        heat_adjustment_s_per_km=0.0,
        citations=[],
    )
    total = pace * 42 + round(pace * 0.2)
    brief = RunnerBrief(goal_time_min=round(total / 60), temp_c=12)
    violations = verify(plan, FLAT_COURSE, brief)
    assert any("citations" in v for v in violations)


def test_heat_adjustment_out_of_band_flagged():
    pace = 341
    plan = PacingPlan(
        strategy_summary="Even pacing.",
        splits=_flat_splits(pace),
        predicted_finish="4:00:00",
        fade_allowance_pct=0.0,
        heat_adjustment_s_per_km=0.0,  # should be > 0 given hot temp
        citations=["heat-tax"],
    )
    total = pace * 42 + round(pace * 0.2)
    brief = RunnerBrief(goal_time_min=round(total / 60), temp_c=28)
    violations = verify(plan, FLAT_COURSE, brief)
    assert any("heat_adjustment_s_per_km" in v for v in violations)


def test_uncoverage_flagged():
    plan = PacingPlan(
        strategy_summary="Even pacing.",
        splits=_flat_splits(341)[:-1],  # drop the partial final row
        predicted_finish="4:00:00",
        fade_allowance_pct=0.0,
        heat_adjustment_s_per_km=0.0,
        citations=["even-pacing-strategy"],
    )
    brief = RunnerBrief(goal_time_min=240, temp_c=12)
    violations = verify(plan, FLAT_COURSE, brief)
    assert any("coverage" in v.lower() for v in violations)
