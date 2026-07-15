# My GLM-5.2 work on 1 DGX Spark (GB10)

**Vincent Marquez** · 2026  

Personal repo: what I ran, measured, and contributed while working with **GLM-5.2** (744B MoE) on an **NVIDIA DGX Spark · GB10 · 121 GB unified memory**.

---

## Tribute — colibrì by Justin (JustVugg)

This work stands on **[colibrì](https://github.com/JustVugg/colibri)** by **[JustVugg](https://github.com/JustVugg)** and contributors.

colibrì is the tiny C engine that makes huge MoE models practical on a single box (streaming experts from disk, optional GPU path, great community tooling).  

**All credit for the core project belongs there.**  
I am a user/contributor who measured and extended pieces of it on GB10 — not the author of colibrì.

| Upstream | Link |
|----------|------|
| Project | https://github.com/JustVugg/colibri |
| License | **Apache License 2.0** |
| Author | JustVugg + community |

Thank you, Justin and everyone who built colibrì.

---

## What I did

| Work | Where |
|------|--------|
| Ran **GLM-5.2 int4** full **top-8** on GB10; measured **decode tok/s** in honest tiers | [docs/NUMBERS.md](docs/NUMBERS.md) |
| Designed / implemented **CACHE_ROUTE** (opt-in cache-aware MoE routing, default **off**) | Merged upstream: **[PR #199](https://github.com/JustVugg/colibri/pull/199)** |
| Explained the mechanism on the project issue | [#161 comment](https://github.com/JustVugg/colibri/issues/161#issuecomment-4970926845) |
| Wrote a **quality A/B harness** with live progress (stop-safe) | [`scripts/quality-ab-cr.py`](scripts/quality-ab-cr.py) |
| Documented hardware, flags, and lab results | [`docs/`](docs/) |

My **engine code contribution** lives in upstream after merge (commit `62419af` via PR #199).  
This repo keeps **notes, scripts, and results** so others can see the GB10 story in one place.

Fork of the engine (for history / local branches): https://github.com/VincentMarquez/colibri  

---

## Headline numbers (labeled)

**Metric:** decode tok/s · **width:** full K=8 · **machine:** this GB10  

| Tier | Setup | decode tok/s | notes |
|------|--------|-------------:|-------|
| **A** | Stock routing | **~2.1–2.4** | fair full-k8 baseline |
| **B** | + experimental **CACHE_ROUTE** (early stack) | **~3.33** | hit ~97%, ~14% route swap |
| **C** | CR + later CUDA stack (MLA / fuse / device-tier) | **~5.1–6.2** | **not CR alone** |

**Disk (flag loudly on NVMe):** iobench **4.25 GB/s** buffered → **9.69 GB/s O_DIRECT** (**2.3×**).  
On a real NVMe that is the page-cache tax; use **`DIRECT=1`** for fair decode runs.  
On slow/VHDX-backed disks O_DIRECT is a weaker lever — measure both (see [docs/HARDWARE.md](docs/HARDWARE.md)).

Details: [docs/NUMBERS.md](docs/NUMBERS.md) · [docs/HARDWARE.md](docs/HARDWARE.md) · [docs/CACHE_ROUTE.md](docs/CACHE_ROUTE.md)

---

## CACHE_ROUTE (my main engine contribution)

Opt-in max-rank style routing (idea related to [arXiv:2412.00099](https://arxiv.org/abs/2412.00099)):

1. Router still scores all experts every layer.  
2. Always take true top-**J** (default 2).  
3. Fill remaining slots preferring experts already **resident** (pin ∪ LRU) within top-**M**.

```bash
# default: stock routing
CACHE_ROUTE=0

# experimental
CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=12
```

**Default remains off** in upstream. Complementary to **PILOT** (prefetch without changing expert IDs).

Shipped in: https://github.com/JustVugg/colibri/pull/199  

---

## Scripts in this repo

```bash
# Quality A/B harness (live req/s · ~tok/s · running accuracy; Ctrl-C keeps partial results)
export COLI_MODEL=/path/to/glm52-colibri-int4
export GLM_BIN=/path/to/colibri/c/glm
export BENCH_DATA=/path/to/colibri/c/bench

./scripts/quality-ab-cr.sh --limit 20 --tasks hellaswag
```

See [scripts/README.md](scripts/README.md) and [docs/QUALITY.md](docs/QUALITY.md).

---

## Repo layout

```
README.md              ← you are here (tribute + my work)
NOTICE                 ← Apache attribution for upstream
docs/HARDWARE.md
docs/NUMBERS.md
docs/CACHE_ROUTE.md
docs/QUALITY.md
docs/CONTRIBUTIONS.md
scripts/quality-ab-cr.py   ← my harness code
scripts/quality-ab-cr.sh
results/                   ← lab snapshots
```

---

## License

| Content | License |
|---------|---------|
| **colibrì engine** (upstream / my PR into it) | **Apache-2.0** — see [JustVugg/colibri](https://github.com/JustVugg/colibri) |
| **Scripts & notes in this repo** (authored by me) | **MIT** — [LICENSE](LICENSE) |

When you use the engine, follow Apache-2.0 and credit **colibrì / JustVugg**.

---

## How to cite

```text
Vincent Marquez — GLM-5.2 lab work on NVIDIA DGX Spark (GB10), 2026
https://github.com/VincentMarquez/glm52-gb10-colibri

Built with colibrì by JustVugg:
https://github.com/JustVugg/colibri

CACHE_ROUTE contribution:
https://github.com/JustVugg/colibri/pull/199
```
