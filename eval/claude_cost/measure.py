"""API-equivalent cost and latency of the Claude teacher column (2026-11-05).

Calls training/bench_copilot.run_teacher, unchanged, on the first N scenarios of the
held-out test set (load_test_scenarios order, the same order the benchmark used), so
the prompt and CLI flags are exactly the benchmark's:
`claude -p --model sonnet --output-format json`, run from the repo root. Each plan
is parsed and checked with the same verifier.
subprocess.run is wrapped only to keep total_cost_usd, duration_ms and token usage
from the CLI's JSON reply, which llm.py drops.

Writes claude_calls.csv and summary.json next to this file.
Usage: python3 eval/claude_cost/measure.py
"""
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "training"))

RECORDS: list[dict] = []
_real_run = subprocess.run


def _recording_run(args, *a, **kw):
    """Pass every call through unchanged; when it is the Claude CLI, keep the
    cost, duration and token fields of its --output-format json reply."""
    t0 = time.perf_counter()
    proc = _real_run(args, *a, **kw)
    wall = time.perf_counter() - t0
    if isinstance(args, list) and args and args[0] == "claude":
        rec = {"wall_s": round(wall, 2), "args": " ".join(args[1:]), "cwd": str(kw.get("cwd") or ".")}
        try:
            out = json.loads(proc.stdout)
            u = out.get("usage", {})
            rec.update({
                "duration_ms": out.get("duration_ms"),
                "cost_usd": out.get("total_cost_usd"),
                "input_tokens": u.get("input_tokens"),
                "output_tokens": u.get("output_tokens"),
                "cache_read": u.get("cache_read_input_tokens"),
                "cache_write": u.get("cache_creation_input_tokens"),
                "model": "+".join(sorted(out.get("modelUsage", {}).keys())),
                "is_error": out.get("is_error"),
            })
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)[:200]
        RECORDS.append(rec)
    return proc


subprocess.run = _recording_run


def pct(values, q):
    """Nearest-rank percentile."""
    v = sorted(values)
    k = max(0, min(len(v) - 1, math.ceil(q / 100 * len(v)) - 1))
    return v[k]


def machine():
    chip = _real_run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return f"{chip}, macOS {platform.mac_ver()[0]}"


def write(rows, extra):
    with open(HERE / "claude_calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    lats = [r["item_wall_s"] for r in rows]
    s = {
        "date": date.today().isoformat(), "machine": machine(),
        "percentile_method": "nearest rank", "n": len(rows),
        "models": sorted({r.get("model") or "?" for r in rows}),
        "cost_per_item_usd_median": round(statistics.median(costs), 4),
        "cost_per_item_usd_mean": round(statistics.mean(costs), 4),
        "cost_per_1000_items_usd": round(1000 * statistics.mean(costs), 2),
        "latency_median_s": round(statistics.median(lats), 2),
        "latency_p95_s": round(pct(lats, 95), 2),
        "cost_note": "total_cost_usd from claude -p --output-format json: the API-equivalent "
                     "cost; the CLI runs on a Max subscription, so nothing was billed",
        **extra,
    }
    (HERE / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))



N = 20


def main():
    import bench_copilot as B
    items = B.load_test_scenarios()[:N]
    out_rows = []
    for i, it in enumerate(items):
        n0, t0 = len(RECORDS), time.perf_counter()
        try:
            raw, _ = B.run_teacher(it["prompt"])
            err = None
        except Exception as e:  # noqa: BLE001
            raw, err = "", f"{type(e).__name__}: {e}"[:200]
        item_wall = time.perf_counter() - t0
        rec = RECORDS[-1] if len(RECORDS) > n0 else {}
        plan, perr = B.parse_plan(raw) if raw else (None, err)
        viol = B.verify(plan, it["course"], it["brief"], retrieved_ids=it["retrieved_ids"]) if plan else None
        out_rows.append({"item_id": it["item_id"], "parsed": plan is not None,
                         "verifier_pass": plan is not None and not viol,
                         "error": err or "", "item_wall_s": round(item_wall, 2),
                         "calls": len(RECORDS) - n0,
                         **{k: v for k, v in rec.items() if k != "args"}})
        print(i, out_rows[-1]["verifier_pass"], rec.get("cost_usd"), round(item_wall, 1), flush=True)
    write(out_rows, {"task": "write a per-km race pacing plan for one course and runner brief",
                     "sample": f"first {N} of the 70 held-out scenarios",
                     "cli_args": RECORDS[0]["args"] if RECORDS else "", "cwd": "repo root",
                     "parsed": sum(r["parsed"] for r in out_rows),
                     "verifier_pass": sum(r["verifier_pass"] for r in out_rows)})


if __name__ == "__main__":
    main()
