# Race Day Copilot — 3-way benchmark

Held-out test set: **70 scenarios** (split by scenario id, seed=42, 90/5/5).

## Objective: parse success & verifier pass rate (zero-shot, no retry)

| System | Parse success | Verifier pass rate | Mean latency (s/plan) |
|---|---|---|---|
| base | 55.7% | 0.0% | 17.50 |
| lora | 95.7% | 0.0% | 53.66 |
| teacher | 90.0% | 75.7% | 157.68 |

## Per-check violation rates (share of items failing each check)

| System | citations | coverage | fade | heat | parse | plausibility | sum |
|---|---|---|---|---|---|---|---|
| base | 0.0% | 55.7% | 0.0% | 0.0% | 44.3% | 0.0% | 0.0% |
| lora | 0.0% | 2.9% | 91.4% | 4.3% | 4.3% | 1.4% | 82.9% |
| teacher | 8.6% | 0.0% | 5.7% | 0.0% | 10.0% | 0.0% | 2.9% |

## LLM judge: LoRA vs teacher (blind, position-swapped)

N judged: 60

- Student (LoRA) preferred: 0.0%
- Teacher preferred: 93.3%
- Tie: 6.7%