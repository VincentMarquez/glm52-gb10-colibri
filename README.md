# GLM-5.2 on NVIDIA DGX Spark (GB10) · colibrì notes

**Author:** [VincentMarquez](https://github.com/VincentMarquez)  
**Upstream engine:** [JustVugg/colibri](https://github.com/JustVugg/colibri)  
**Date:** 2026-07

Personal lab notes and measurements for running **GLM-5.2 (744B MoE, int4 streaming)** with **colibrì** on an **NVIDIA DGX Spark · GB10 · 121 GB unified memory**.

This is **not** a fork of the whole engine. It is:

- Honest **decode tok/s** numbers (full top‑8 width)
- An experimental **CACHE_ROUTE** idea (opt‑in, default off)
- A **routing-only PR** against upstream
- Scripts for **quality A/B** with live progress

---

## Headline numbers (read the labels)

All numbers: **decode tok/s**, **full expert width K=8** (`TOPK` unset), same GB10 box unless noted.

| Tier | Setup | decode tok/s | hit | notes |
|------|--------|-------------:|----:|-------|
| **A** | Stock routing | **~2.1–2.4** | ~82% | fair full‑k8 baseline |
| **B** | + experimental **CACHE_ROUTE** (early stack) | **~3.33** | **97%** | ~**14%** expert swap vs true top‑8 |
| **C** | CACHE_ROUTE + later CUDA stack (MLA / fuse / device‑tier) | **~5.1–6.2** | ~94–97% | **not CACHE_ROUTE alone** |

**Disk (iobench):** 4.25 GB/s buffered · **9.69 GB/s O_DIRECT**.

> **Do not cite “6 tok/s” without Tier C context.**  
> The clean routing-side story is **~2.4 → 3.33** (Tier A → B).  
> The **5–6** band needs residency + fused CUDA path **and** CACHE_ROUTE.

Details: [docs/NUMBERS.md](docs/NUMBERS.md) · [docs/HARDWARE.md](docs/HARDWARE.md)

---

## What I contributed

| Item | Link / status |
|------|----------------|
| Issue #161 discussion (mechanism + tiered numbers) | [comment](https://github.com/JustVugg/colibri/issues/161#issuecomment-4970926845) |
| **PR: opt‑in CACHE_ROUTE** (routing-only, default off) | [JustVugg/colibri#199](https://github.com/JustVugg/colibri/pull/199) |
| Quality A/B harness (live tok/s + partial results on stop) | [`scripts/quality-ab-cr.py`](scripts/quality-ab-cr.py) |

Upstream remains Justin’s project. Credit for colibrì itself belongs to the [colibri](https://github.com/JustVugg/colibri) maintainers and community.

---

## CACHE_ROUTE in one paragraph

Inspired by [arXiv:2412.00099](https://arxiv.org/abs/2412.00099) (max-rank style):

1. Router still scores **all** experts every layer (no skip, no reuse of previous top‑k).
2. Always take true top‑**J** (default **2**), even if cold on disk.
3. Fill remaining slots preferring experts already **resident (pin ∪ LRU)** that still rank in top‑**M** (default **12**, also **16**).

**Default: OFF.** Complementary to **PILOT** (prefetch without changing expert IDs). CACHE_ROUTE *can* change which experts run (~14–22% slot swap).

Flags: [docs/CACHE_ROUTE.md](docs/CACHE_ROUTE.md)

```bash
CACHE_ROUTE=0 ./coli chat
CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=12 ./coli chat
```

---

## Quality (limited sample)

Log-likelihood SCORE harness (not chat decode). HellaSwag subset, n=20 questions.

| Cell | Status | acc_norm |
|------|--------|----------|
| Stock (`CACHE_ROUTE=0`) | **complete** | **80%** (n=20) |
| CACHE_ROUTE on | in progress / see snapshot | TBD vs stock |

Raw `acc` on HellaSwag is noisy; harness prefers **acc_norm**. Small *n* — not a full `./coli bench` gate.  
Snapshot: [results/quality_ab_snapshot.json](results/quality_ab_snapshot.json) · notes: [docs/QUALITY.md](docs/QUALITY.md)

```bash
# live progress; Ctrl-C keeps partial results
./scripts/quality-ab-cr.sh --limit 20 --tasks hellaswag
tail -f ~/quality-ab-cr/latest/cr_off/STATUS.txt
```

---

## Hardware (short)

| | |
|--|--|
| Machine | NVIDIA **DGX Spark** (`NVIDIA_DGX_Spark`) |
| GPU / mem | **GB10** · **121 GB** unified |
| CPU | 20× Arm (10× X925 + 10× A725) |
| OS | Ubuntu 24.04 · Linux aarch64 |

Full sheet: [docs/HARDWARE.md](docs/HARDWARE.md)

---

## Repo map

```
README.md                 ← you are here
docs/HARDWARE.md
docs/NUMBERS.md           ← full tier tables + caveats
docs/CACHE_ROUTE.md       ← mechanism + env flags
docs/QUALITY.md           ← quality protocol + results
docs/CONTRIBUTIONS.md     ← PR/issue links, how to cite
scripts/quality-ab-cr.py  ← streaming SCORE A/B
scripts/quality-ab-cr.sh
results/                  ← frozen snapshots
```

---

## How to cite / give credit

If these numbers or CACHE_ROUTE help you:

```text
Vincent Marquez — GLM-5.2 / colibrì notes on DGX Spark (GB10)
https://github.com/VincentMarquez/glm52-gb10-colibri
PR: https://github.com/JustVugg/colibri/pull/199
```

Please also credit **colibrì**: https://github.com/JustVugg/colibri  
Paper idea: arXiv:2412.00099

---

## License

Notes and scripts in this repo: [MIT](LICENSE).  
colibrì engine code remains under its own upstream license.
