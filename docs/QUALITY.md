# Quality protocol and results

Goal: estimate whether **CACHE_ROUTE** hurts multiple-choice log-likelihood quality vs stock routing.

## Method

Upstream-style SCORE path (same idea as `coli bench` / `tools/eval_glm.py`):

- Local JSONL tasks (`hellaswag`, etc.)
- For each question option: full forward, sum log-prob of continuation tokens
- Metrics: **acc** (raw) and **acc_norm** (length-normalized) — **prefer acc_norm**

Harness with **live progress** (req/s, ~tok/s, running accuracy) and **checkpoints on stop**:

```bash
./scripts/quality-ab-cr.sh --limit 20 --tasks hellaswag
# or only one cell:
./scripts/quality-ab-cr.sh --only off --limit 20 --tasks hellaswag
./scripts/quality-ab-cr.sh --only on  --limit 20 --tasks hellaswag
```

Artifacts (default): `~/quality-ab-cr/latest/`

| File | Purpose |
|------|---------|
| `*/STATUS.txt` | live line (tail -f) |
| `*/progress.jsonl` | every completed option |
| `*/checkpoint_*.json` | every N options |
| `*/SUMMARY.txt` | final or partial summary |
| `COMPARE.txt` | off vs on when both exist |

Stop anytime (`Ctrl-C` / kill harness). Partial results remain.

## Results — HellaSwag n=20 (seed=1234) · COMPLETE

Same 20 questions, same request file, full K=8 SCORE mode.

| Cell | Options | Wall | acc | **acc_norm** |
|------|--------:|-----:|----:|-------------:|
| **Stock** `CACHE_ROUTE=0` | 80/80 | ~47 min | 40% | **80%** |
| **CACHE_ROUTE=1** J=2 M=12 | 80/80 | ~47 min | 35% | **70%** |
| **Delta (on − off)** | | | −5 pp | **−10 pp** |

Files:

- [results/COMPARE.txt](../results/COMPARE.txt)
- [results/quality_ab_snapshot.json](../results/quality_ab_snapshot.json)
- [results/cr_off_SUMMARY.json](../results/cr_off_SUMMARY.json)
- [results/cr_on_SUMMARY.json](../results/cr_on_SUMMARY.json)

### How to read this

| OK to say | Not OK to say |
|-----------|----------------|
| “On n=20 HellaSwag SCORE, CR was ~10 pp lower acc_norm” | “CR destroys quality” / “quality free” |
| “Supports keeping CR **opt-in / never default** until larger benches” | “Full MMLU / published HellaSwag proven” |
| “Stock int4 SCORE looks ~80% acc_norm on this tiny slice” | “Matches model-card HellaSwag” |

**n=20 is noisy** (± a few questions moves % a lot). Treat as a **leading indicator**, not a gate to ship as default.

Next step when idle: larger `./coli bench` (`hellaswag,arc_challenge,mmlu` with higher `--limit`).

## Requirements

- Built `glm` binary with SCORE mode (and CACHE_ROUTE if testing `cr_on`)
- Model snap (e.g. int4 colibri directory with `tokenizer.json`)
- Bench JSONL under `c/bench/` or `~/.cache/colibri/bench/`
- Python `tokenizers`
