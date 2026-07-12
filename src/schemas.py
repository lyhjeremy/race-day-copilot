"""Pydantic contracts for Race Day Copilot. See RACE_DAY_COPILOT_SPEC.md §1."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Segment(BaseModel):
    km_start: float
    km_end: float
    trend: Literal["up", "down", "flat"]
    severity: Literal["gentle", "moderate", "steep"]
    note: str | None = None


class CourseProfile(BaseModel):
    race_name: str | None = None
    distance_km: float
    segments: list[Segment]
    source_confidence: float


class RunnerBrief(BaseModel):
    goal_time_min: int = Field(ge=75, le=390)
    temp_c: float | None = None
    experience: Literal["first", "1-3", "4+"] | None = None
    recent_race: str | None = None
    units: Literal["km", "mi"] = "km"


class SplitRow(BaseModel):
    km: int
    target_pace: str
    cumulative: str
    note: str | None = None


class PacingPlan(BaseModel):
    strategy_summary: str
    splits: list[SplitRow]
    predicted_finish: str
    fade_allowance_pct: float
    heat_adjustment_s_per_km: float
    cautions: list[str] = []
    citations: list[str] = []
