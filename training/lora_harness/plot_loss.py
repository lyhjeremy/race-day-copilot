"""Parse mlx_lm's training log (loss_lines.txt from train.sh) into a loss-
curve figure. No W&B dependency -- matplotlib only, portfolio convention.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_LINE_RE = re.compile(r"Iter (\d+): (Train|Val) loss ([\d.]+)")


def parse_loss_log(path: str | Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {"Train": [], "Val": []}
    for line in Path(path).read_text().splitlines():
        m = _LINE_RE.search(line)
        if m:
            iters, kind, loss = int(m.group(1)), m.group(2), float(m.group(3))
            series[kind].append((iters, loss))
    return series


def plot(series: dict[str, list[tuple[int, float]]], out_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, pts in series.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=3, label=kind)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    log_path, out_path = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "LoRA training loss"
    plot(parse_loss_log(log_path), out_path, title)
