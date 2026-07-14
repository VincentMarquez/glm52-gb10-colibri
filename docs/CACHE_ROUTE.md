# CACHE_ROUTE — opt-in cache-aware MoE routing

**Default: OFF.** Stock full top‑K router unless `CACHE_ROUTE=1`.

Paper-style max-rank selection: [arXiv:2412.00099](https://arxiv.org/abs/2412.00099).

## What it is **not**

- Not reusing the **previous token’s** top‑k  
- Not skipping the router  
- Not the same as **PILOT** (prefetch)

## What it **is**

Per sparse MoE layer, after usual sigmoid router scores:

1. Rank experts by router score (same logits as stock).
2. **Always take true top‑`ROUTE_J`** (default **2**), even if uncached.
3. Fill remaining **K−J** slots preferring **pin ∪ LRU** experts that still rank inside top‑`ROUTE_M` (default **12**; we also use **16**).
4. If the window cannot fill K, fall back to true ranking order.

Optional:

- `ROUTE_ALPHA` — scale gate mass of *substituted* experts before renorm (`1` = off).
- `ROUTE_AGREE` — overlap % + mean KL vs true top‑K (auto-on when CR is on).

| Lever | Acts on | Changes expert IDs? |
|-------|---------|---------------------|
| PILOT | when cold experts load | **no** |
| CACHE_ROUTE | which experts are selected | **yes** (within top‑M) |
| Memory layout / pin | what ends up resident | no (placement) |

## Does it change output?

**Yes — not greedy-identical to stock top‑8.**

Typical substitution: ~**14–22%** of route slots vs true top‑K; **route_agree ~78–87%** depending on M / warm.

Keep **opt-in / never default** until larger quality gates pass.

## Env flags

| Env | Default | Meaning |
|-----|---------|---------|
| `CACHE_ROUTE` | `0` | enable max-rank cache-aware fill |
| `ROUTE_J` | `2` | sacred top ranks (always taken) |
| `ROUTE_M` | `12` | max-rank window for resident prefer |
| `ROUTE_P` | `0` | optional cumulative mass window |
| `ROUTE_ALPHA` | `1` | scale substituted gate mass |
| `ROUTE_AGREE` | auto | telemetry; auto-on with CR |

```bash
CACHE_ROUTE=0 ./coli chat
CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=12 ./coli chat
CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=16 ./coli chat

# A/B vs PILOT
CACHE_ROUTE=1 PILOT=0 ...
CACHE_ROUTE=0 PILOT=1 ...
```

## Stats to watch

- `swap N%` / `swap_pct` — fraction of chosen slots not in true top‑K  
- `route_swaps` / `route_slots`  
- `route_agree` — |chosen ∩ true top‑K| / K  
- `route_kl`  
- `hit N%` — expert cache residency (disk)

## Upstream PR

Routing-only port (no full GB10 CUDA stack):

**https://github.com/JustVugg/colibri/pull/199**

Mechanism write-up on the issue:

**https://github.com/JustVugg/colibri/issues/161#issuecomment-4970926845**
