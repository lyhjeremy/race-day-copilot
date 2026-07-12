"""Draw example elevation-profile images with matplotlib (zero copyright
risk) for the app's example gallery AND as vision-extraction test fixtures.
Also emits the ground-truth CourseProfile JSON for each, used by
gen_scenarios.py's scenario grid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from schemas import CourseProfile, Segment

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "example_courses"

ARCHETYPES = {
    "flat_berlin_like": {
        "distance_km": 42.2,
        "elevation_fn": lambda km: 40 + 5 * np.sin(km / 8),
        "segments": [Segment(km_start=0, km_end=42.2, trend="flat", severity="gentle",
                              note="Flat, fast course")],
    },
    "rolling": {
        "distance_km": 42.2,
        "elevation_fn": lambda km: 60 + 25 * np.sin(km / 4),
        "segments": [
            Segment(km_start=0, km_end=10, trend="up", severity="gentle"),
            Segment(km_start=10, km_end=20, trend="down", severity="gentle"),
            Segment(km_start=20, km_end=30, trend="up", severity="moderate"),
            Segment(km_start=30, km_end=42.2, trend="down", severity="gentle"),
        ],
    },
    "hilly_boston_like": {
        "distance_km": 42.2,
        "elevation_fn": lambda km: 50 - 0.5 * km + (40 if 26 <= km <= 34 else 0) * np.sin((km - 26) / 8 * np.pi),
        "segments": [
            Segment(km_start=0, km_end=16, trend="down", severity="gentle"),
            Segment(km_start=16, km_end=26, trend="flat", severity="gentle"),
            Segment(km_start=26, km_end=34, trend="up", severity="steep", note="Newton hills"),
            Segment(km_start=34, km_end=42.2, trend="down", severity="moderate"),
        ],
    },
    "net_downhill": {
        "distance_km": 42.2,
        "elevation_fn": lambda km: 300 - 6 * km,
        "segments": [Segment(km_start=0, km_end=42.2, trend="down", severity="moderate",
                              note="Net downhill point-to-point")],
    },
    "out_and_back_climb": {
        "distance_km": 42.2,
        "elevation_fn": lambda km: 20 * abs(np.sin(km / 42.2 * np.pi)) * 8,
        "segments": [
            Segment(km_start=0, km_end=21.1, trend="up", severity="moderate"),
            Segment(km_start=21.1, km_end=42.2, trend="down", severity="moderate"),
        ],
    },
    "half_marathon_flat": {
        "distance_km": 21.1,
        "elevation_fn": lambda km: 30 + 3 * np.sin(km / 5),
        "segments": [Segment(km_start=0, km_end=21.1, trend="flat", severity="gentle")],
    },
}


def draw_and_save(name: str, spec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    distance = spec["distance_km"]
    kms = np.linspace(0, distance, 300)
    elevs = np.array([spec["elevation_fn"](k) for k in kms])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(kms, elevs, elevs.min() - 5, alpha=0.3, color="#7a1f2b")
    ax.plot(kms, elevs, color="#7a1f2b", linewidth=1.5)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(name.replace("_", " ").title())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150)
    plt.close(fig)

    profile = CourseProfile(
        race_name=name.replace("_", " ").title(),
        distance_km=distance,
        segments=spec["segments"],
        source_confidence=1.0,  # ground truth, not a vision extraction
    )
    (OUT_DIR / f"{name}.json").write_text(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    for name, spec in ARCHETYPES.items():
        draw_and_save(name, spec)
        print(f"Wrote {name}.png + {name}.json")
