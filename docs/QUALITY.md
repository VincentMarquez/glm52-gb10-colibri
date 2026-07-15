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

## Independent reproduction — x86 dual 5090 (community)

Same idea reimplemented from the public description (not a GB10-only quirk).  
Box: Ryzen 9 9950X3D2 · dual RTX 5090 · Gen5 NVMe · PCIe (no unified memory).  
Reported in [Discussions #208](https://github.com/JustVugg/colibri/discussions/208) / trail #120→#215.

| Config | tok/s (1024-tok window) | hit | substitution |
|--------|------------------------:|----:|-------------:|
| true top-8 | **2.2554** | 79% | — |
| `ROUTE_J=2` `ROUTE_M=12` | **3.09 / 3.04** | 92% | **15.3%** |

Speed: **+42%** there vs **~+39%** on GB10 (2.39 → 3.33) — consistent across architectures.  
Substitution **15.3%** vs our **~14%** — reproduces.

### Quality — MMLU-200 (same build, same seeds, clean learned-cache both cells)

| Cell | MMLU-200 |
|------|---------:|
| true routing | **59.0%** |
| `ROUTE_J=2` `ROUTE_M=12` | **62.0%** |

**Honest read:** binomial noise at n=200 is ~±3.5 pp → **no detectable difference**, not “substitution helps.”  
That is already much stronger than primes eyeball: **~15% expert substitution did not measurably dent MMLU** on this gate. Full bench / `--limit 1000` still welcome.

Discipline: these rows stay **off the main true-routing table** (research / experimental flag only).

## Interpretation (vs EXPERT_BUDGET)

- **EXPERT_BUDGET** (blind drop by gate weight) can speed up but **cliffs** coherence on reasoning prompts.
- **CACHE_ROUTE** substitutes toward **already-resident** experts inside top-M → ~14% swap, quality often holds better (principled middle path; arXiv:2412.00099 style).
- This HellaSwag n=20 cell: stock **80%** → CR **70%** acc_norm (−10 pp), but **85% same pick** — not a free lunch, not a collapse.
- Supports keeping CACHE_ROUTE **opt-in / default off** until larger gates land.
- Connects to expert-specialization work ([colibrì #207](https://github.com/JustVugg/colibri/issues/207)): *which* experts are safe to substitute.

## Next quality gate (requested for ROUTE_J / ROUTE_M tuning)

Preferred over primes-only eyeball:

1. **Greedy-token agreement** vs true top-8 across a **small reasoning set** (not only primes).  
2. Same seed, same prompt file, `CACHE_ROUTE=0` vs `1` with fixed J/M.  
3. Report: % tokens identical, first-divergence position, optional ROUTE_AGREE / swap%.  

Harness today: SCORE/HellaSwag. Greedy agreement script: extend `scripts/quality-ab-cr.py` or use timed `coli` with fixed seed + diff.

Larger `./coli bench` still welcome when idle.
