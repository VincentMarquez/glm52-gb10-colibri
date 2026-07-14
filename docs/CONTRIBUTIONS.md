# Contributions & how to credit

## Upstream (colibrì)

| | |
|--|--|
| Project | [JustVugg/colibri](https://github.com/JustVugg/colibri) |
| What it is | Tiny C engine to run GLM-5.2 MoE locally with streaming experts |

This notes repo does **not** replace upstream. Features should land via PRs when possible.

## My work (Vincent Marquez)

| Deliverable | Where |
|-------------|--------|
| CACHE_ROUTE mechanism explanation + tiered GB10 numbers | [Issue #161 comment](https://github.com/JustVugg/colibri/issues/161#issuecomment-4970926845) |
| Routing-only **CACHE_ROUTE** PR (default off + telemetry) | [PR #199](https://github.com/JustVugg/colibri/pull/199) |
| Lab notes, tiers, quality harness | **This repo** |

### PR #199 (summary)

- Opt-in max-rank routing (`CACHE_ROUTE`, `ROUTE_J/M/P/ALPHA`, `ROUTE_AGREE`)
- Footer / STAT: swap, agree, kl
- Docs upstream: `docs/CACHE_ROUTE.md` on the PR branch
- **Not** included: full GB10 CUDA fuse / device-tier stack (so maintainers can A/B routing vs PILOT cleanly)

Merge may require **maintainer approval** (branch protection) — not a sign the PR is broken.

## Suggested citation

```text
Vincent Marquez. GLM-5.2 / colibrì measurements on NVIDIA DGX Spark (GB10).
https://github.com/VincentMarquez/glm52-gb10-colibri (2026).
CACHE_ROUTE PR: https://github.com/JustVugg/colibri/pull/199
```

Paper inspiration for max-rank cache-aware routing:

```text
arXiv:2412.00099
```

## Related upstream issues (context)

- Community speed / hardware rows (README + issues)
- PILOT / prefetch discussions (e.g. cross-layer coupling #176 era)
- Routing experiments discussion: **#161**

## What I am **not** claiming

- Ownership of colibrì  
- That 6 tok/s is portable or default  
- That CACHE_ROUTE is quality-proven at full bench scale  
- That TOPK prune equals full top‑8 quality  
