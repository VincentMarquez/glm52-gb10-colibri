# Draft — JustVugg/colibri Discussions #208 (Show and tell)

Post to: https://github.com/JustVugg/colibri/discussions/208

---

## DGX Spark / GB10 — stock top-8 vs CACHE_ROUTE (+ O_DIRECT disk)

Two-tier discipline: **stock full top-8** vs experimental **CACHE_ROUTE**. Metric is **decode tok/s** (generation window), not overall wall tok/s.

```
Hardware:  NVIDIA DGX Spark · GB10 (sm_121) / 20× ARM (X925+A725) / 121 GB unified
           local NVMe · iobench 4.25 GB/s buffered · 9.69 GB/s O_DIRECT (2.3×)
OS:        Ubuntu 24.04 · Linux 6.17 · aarch64 · native (not WSL)
Commit:    CACHE_ROUTE via PR #199 (62419af); engine stack as measured in notes
Model:     GLM-5.2 int4 (colibrì snap) · full K=8 · MTP off for speed cells
Command:   DIRECT=1 + CUDA unified/dense path · large pin/LRU
           Stock:  CACHE_ROUTE=0
           Exp:    CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=12  (optional ROUTE_AGREE=1)
Warm-up:   session warm LRU / multi-turn chat with :reset; timed PROFILE WARM=160 NGEN=48
Result (decode tok/s · full K=8):

  Tier A stock routing:     2.08–2.39 decode · hit ~82%
  Tier B CACHE_ROUTE early: 3.33 decode · hit 97% · swap 14%   (chat ladder peak)
                            2.67 decode · hit 93.5% · swap 15% (timed one-shot)
  Tier C later CUDA stack + CR (not CR alone): 5.14–6.17 decode band
                            fair CR on/off same stack: 2.98 → 5.14

Disk lever: DIRECT=1 / O_DIRECT 9.69 GB/s vs 4.25 buffered — flag louder on real NVMe;
            weaker lever on slow/VHDX-backed disks (measure both).
```

### Why two tiers

| Tier | What it isolates |
|------|------------------|
| **A** | Leaderboard-comparable **stock** full-k8 routing |
| **B** | **CACHE_ROUTE alone** on early stack (~2.4 → 3.33) |
| **C** | CR **plus** MLA/fuse/device-tier — do **not** call this “CR alone” |

### Quality A/B (limited — not full coli bench)

Protocol: SCORE log-likelihood · HellaSwag · **n=20** · seed 1234 · same request file · full K=8.

| Cell | acc_norm | acc | notes |
|------|---------:|----:|-------|
| `CACHE_ROUTE=0` | **80%** (16/20) | 40% | stock |
| `CACHE_ROUTE=1` J=2 M=12 | **70%** (14/20) | 35% | −10 pp |
| Same normalized pick | **85%** | | 2 flips stock✓→CR✗ |

Caveat: **n=20 only**. Primes multi-turn still produced correct first-20 list (eyeball). Prefer larger greedy-token agreement set for ROUTE_J/M defaults. Supports **default off**.

Vs **EXPERT_BUDGET** (blind drop): CR is the principled middle path — substitute toward **resident** experts inside top-M (arXiv:2412.00099), not drop by gate weight alone. Connects to expert-specialization (#207): which experts are safe to swap.

### Links

- Notes / numbers: https://github.com/VincentMarquez/glm52-gb10-colibri  
- CACHE_ROUTE PR: https://github.com/JustVugg/colibri/pull/199  
- Mechanism comment: https://github.com/JustVugg/colibri/issues/161#issuecomment-4970926845  
- Quality snapshot JSON in that repo under `results/`

Happy to extend with greedy-token agreement on a small reasoning set (not just primes) when the box is free — that should help tune `ROUTE_J` / `ROUTE_M`.
