# Contributions

## Upstream project (tribute)

Everything here depends on **[colibrì](https://github.com/JustVugg/colibri)** by **JustVugg** and contributors (Apache-2.0).

I did not create colibrì. I used it, measured it on GB10, and contributed **CACHE_ROUTE**.

## My upstream contribution

| Item | Link |
|------|------|
| **PR #199** — opt-in CACHE_ROUTE (default off) | https://github.com/JustVugg/colibri/pull/199 |
| Status | **Merged** 2026-07-14 |
| Merge commit | `62419af1884af321458472f6927c63ab07f67427` |
| Mechanism discussion | https://github.com/JustVugg/colibri/issues/161#issuecomment-4970926845 |

### What PR #199 contains

- `CACHE_ROUTE` / `ROUTE_J` / `ROUTE_M` / `ROUTE_P` / `ROUTE_ALPHA` / `ROUTE_AGREE`
- Selection path in MoE routing (max-rank, pin∪LRU prefer)
- Footer / STAT telemetry (`swap`, `agree`, `kl`)
- Short upstream docs for the flags  

Routing-only; not the full GB10 CUDA stack.

## Code in *this* repo (mine)

| Path | Description |
|------|-------------|
| `scripts/quality-ab-cr.py` | Streaming SCORE A/B harness (progress, checkpoints) |
| `scripts/quality-ab-cr.sh` | Wrapper |
| `docs/*` | Hardware, numbers, flags, quality notes |
| `results/*` | Frozen lab snapshots |

## Engine code

Prefer **upstream main** (or a release after #199) for CACHE_ROUTE:

```bash
git clone https://github.com/JustVugg/colibri.git
# build per upstream README
```

My fork (optional, for history): https://github.com/VincentMarquez/colibri
