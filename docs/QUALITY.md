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

## Results snapshot (HellaSwag, n=20, seed=1234)

Captured during the first full A/B campaign (see `results/quality_ab_snapshot.json` for timestamps).

### Stock — `CACHE_ROUTE=0` (complete)

| | |
|--|--|
| Options scored | **80 / 80** |
| Wall | ~47 min (~2.5 SCORE “tok/s” — full forwards, not chat decode) |
| **acc** | **40%** (8/20) |
| **acc_norm** | **80%** (16/20) |

Raw HellaSwag accuracy often looks weak without length normalization; the harness standard is **acc_norm**.

### CACHE_ROUTE=1 (partial during first capture)

Early mid-run sample was small (*n* &lt; 20). **Do not conclude** CR quality until `cr_on` finishes the same 20 questions and `COMPARE.txt` is written.

Re-run or refresh:

```bash
# after a finished dual run
cat ~/quality-ab-cr/latest/COMPARE.txt
cp ~/quality-ab-cr/latest/COMPARE.txt results/
```

## Interpretation guardrails

| OK to say | Not OK to say |
|-----------|----------------|
| “On n=20 HellaSwag SCORE, stock acc_norm was 80%” | “Quality-free / matches published GLM HellaSwag” |
| “CR A/B still running / delta = X on n=20” | “Full MMLU proven” |
| “Experimental; default remains off” | “Safe to default CACHE_ROUTE=1” |

Full gate remains: larger `./coli bench` limits (hellaswag + arc + mmlu) when idle.

## Requirements

- Built `glm` binary with SCORE mode (and CACHE_ROUTE if testing `cr_on`)
- Model snap (e.g. int4 colibri directory with `tokenizer.json`)
- Bench JSONL under `c/bench/` or `~/.cache/colibri/bench/`
- Python `tokenizers`
