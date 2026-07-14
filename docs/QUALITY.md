# Quality lab notes (this repo only)

Limited SCORE A/B on HellaSwag to see if CACHE_ROUTE changes multiple-choice log-likelihood.

## Method

Same idea as colibrì `coli bench` / `eval_glm.py`:

- Score each answer option with a full forward (log-likelihood)
- Prefer **acc_norm** (length-normalized)

Harness (my code):

```bash
./scripts/quality-ab-cr.sh --limit 20 --tasks hellaswag
```

Live progress; Ctrl-C keeps partial results under `~/quality-ab-cr/`.

## Snapshot — HellaSwag n=20 (seed 1234)

| Cell | acc_norm | acc |
|------|---------:|----:|
| Stock `CACHE_ROUTE=0` | **80%** (16/20) | 40% |
| `CACHE_ROUTE=1` J=2 M=12 | **70%** (14/20) | 35% |
| Delta | **−10 pp** | −5 pp |

Only **2 questions** flipped stock-correct → CR-wrong; **85%** same normalized pick.  
**n=20 is tiny** — leading indicator, not a full bench gate.

Artifacts: [results/COMPARE.txt](../results/COMPARE.txt), [results/quality_ab_snapshot.json](../results/quality_ab_snapshot.json).

## Interpretation

- Supports keeping CACHE_ROUTE **opt-in / default off** (as merged upstream).
- Does **not** prove “quality free” or “broken.”
- Larger `./coli bench` still welcome when idle.
