"""Task JSONL -> mlx-lm chat-format train/valid/test splits.

Splits by a configurable entity key (not by row) to prevent leakage --
e.g. all rows from the same wine, scenario, or merchant land in the same
split. Emits a dataset card with counts, class balance (if a label_key is
given), and a split-leakage assertion.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def _entity_bucket(entity_key: str, n_buckets: int = 1000) -> int:
    h = hashlib.sha256(entity_key.encode("utf-8")).hexdigest()
    return int(h, 16) % n_buckets


def prep_dataset(records: list[dict], *,
                  entity_key_fn,
                  to_messages_fn,
                  out_dir: str | Path,
                  label_key: str | None = None,
                  split_ratios: tuple[float, float, float] = (0.90, 0.05, 0.05),
                  seed: int = 42) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ratio, valid_ratio, _test_ratio = split_ratios
    train_cut = int(train_ratio * 1000)
    valid_cut = train_cut + int(valid_ratio * 1000)

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    entity_to_split: dict[str, str] = {}

    for rec in records:
        entity = entity_key_fn(rec)
        if entity not in entity_to_split:
            bucket = _entity_bucket(f"{seed}:{entity}")
            if bucket < train_cut:
                entity_to_split[entity] = "train"
            elif bucket < valid_cut:
                entity_to_split[entity] = "valid"
            else:
                entity_to_split[entity] = "test"
        splits[entity_to_split[entity]].append(rec)

    for name, recs in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                messages = to_messages_fn(rec)
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    # Leakage assertion: no entity appears in more than one split.
    entities_by_split = {
        name: {entity_key_fn(r) for r in recs} for name, recs in splits.items()
    }
    overlap = (
        (entities_by_split["train"] & entities_by_split["valid"])
        | (entities_by_split["train"] & entities_by_split["test"])
        | (entities_by_split["valid"] & entities_by_split["test"])
    )
    assert not overlap, f"Split leakage detected for entities: {list(overlap)[:5]}"

    card = {
        "counts": {name: len(recs) for name, recs in splits.items()},
        "n_entities": {name: len(ents) for name, ents in entities_by_split.items()},
        "leakage_check": "passed",
    }
    if label_key:
        card["label_balance"] = {
            name: dict(Counter(r.get(label_key) for r in recs))
            for name, recs in splits.items()
        }

    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return card
