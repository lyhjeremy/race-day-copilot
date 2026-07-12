"""3-way benchmark harness: base model (zero/few-shot floor) vs base+LoRA
adapter (the fine-tuning lift) vs Claude teacher (frontier reference).

Every project supplies its own per-item metric function(s); this module
handles running all three systems over a held-out set and producing a
uniform benchmark.md + CSV with latency/cost always included.
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class SystemResult:
    system: str  # "base" | "lora" | "teacher"
    outputs: list[str]
    latencies_s: list[float]

    @property
    def mean_latency_s(self) -> float:
        return sum(self.latencies_s) / len(self.latencies_s) if self.latencies_s else 0.0


def run_mlx_generate(model: str, prompts: list[str], *, adapter_path: str | None = None,
                      max_tokens: int = 800) -> SystemResult:
    """Run mlx_lm.generate over a list of prompts, one process per call (mlx_lm
    doesn't expose a stable batched Python API across versions -- CLI is the
    honest, version-stable interface). Returns per-item latency for the
    tok/s and s/item benchmark columns.
    """
    outputs, latencies = [], []
    for prompt in prompts:
        args = ["mlx_lm.generate", "--model", model, "--prompt", prompt,
                "--max-tokens", str(max_tokens)]
        if adapter_path:
            args += ["--adapter-path", adapter_path]
        start = time.time()
        proc = subprocess.run(args, capture_output=True, text=True, timeout=180)
        latencies.append(time.time() - start)
        outputs.append(proc.stdout.strip())
    system = "lora" if adapter_path else "base"
    return SystemResult(system=system, outputs=outputs, latencies_s=latencies)


def run_teacher(prompts: list[str], *, tier: str = "smart") -> SystemResult:
    import llm
    outputs, latencies = [], []
    for prompt in prompts:
        resp = llm.generate(prompt, tier=tier)
        outputs.append(resp.text)
        latencies.append(resp.latency_s)
    return SystemResult(system="teacher", outputs=outputs, latencies_s=latencies)


@dataclass
class BenchmarkReport:
    rows: list[dict] = field(default_factory=list)  # one row per system, metric columns + latency

    def add(self, system: str, metrics: dict, mean_latency_s: float, cost_per_1k: float) -> None:
        self.rows.append({
            "system": system, **metrics,
            "latency_s_per_item": round(mean_latency_s, 3),
            "cost_per_1k_calls_usd": cost_per_1k,
        })

    def to_markdown(self, title: str) -> str:
        if not self.rows:
            return f"# {title}\n\n(no rows)"
        cols = list(self.rows[0].keys())
        lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in self.rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)

    def to_csv(self, path: str | Path) -> None:
        if not self.rows:
            return
        cols = list(self.rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(self.rows)

    def save(self, out_dir: str | Path, title: str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "benchmark.md").write_text(self.to_markdown(title))
        self.to_csv(out_dir / "benchmark.csv")


def blind_pairwise_judge(item_a: str, item_b: str, rubric: str) -> str:
    """Judge two outputs blind, position-swapped (call twice, A/B then B/A;
    count a win only if both orderings agree, else 'tie'). Returns
    'a' | 'b' | 'tie'. Caller is responsible for feeding (student, teacher)
    and (teacher, student) and reconciling.
    """
    import llm
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
    """Position-swap-verified pairwise judge. Returns 'student' | 'teacher' | 'tie'."""
    v1 = blind_pairwise_judge(student_output, teacher_output, rubric)  # A=student, B=teacher
    v2 = blind_pairwise_judge(teacher_output, student_output, rubric)  # A=teacher, B=student

    v1_pick = {"a": "student", "b": "teacher", "tie": "tie"}[v1]
    v2_pick = {"a": "teacher", "b": "student", "tie": "tie"}[v2]

    if v1_pick == v2_pick:
        return v1_pick
    return "tie"  # orderings disagree -> treat as a tie, don't overclaim
