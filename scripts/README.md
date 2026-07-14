# Scripts

## `quality-ab-cr.sh` / `quality-ab-cr.py`

CACHE_ROUTE quality A/B with **live progress** (req/s, ~tok/s, running accuracy).  
Stop anytime; partial `progress.jsonl` + checkpoints remain.

```bash
export COLI_MODEL=/path/to/glm52-colibri-int4
export GLM_BIN=/path/to/colibri/c/glm
export BENCH_DATA=/path/to/colibri/c/bench   # hellaswag.jsonl etc.

./quality-ab-cr.sh --limit 20 --tasks hellaswag
# tail -f ~/quality-ab-cr/latest/cr_off/STATUS.txt
```

Requires: Python 3, `tokenizers`, SCORE-capable `glm` binary.

Env overrides: `COLI_MODEL`, `GLM_BIN`, `BENCH_DATA`, `QUALITY_AB_DIR`.
