# 11.1+ tok/s — GLM-5.2 on One NVIDIA DGX Spark (Full Top-8 MoE)

![Peak decode](https://img.shields.io/badge/peak_decode-11.7_mid_tok%2Fs-brightgreen)
![Overall decode](https://img.shields.io/badge/overall_decode-11.1%2B_tok%2Fs-success)
![Full top-8](https://img.shields.io/badge/MoE-full_top--8-blue)
![NVIDIA DGX Spark](https://img.shields.io/badge/hardware-1x_NVIDIA_DGX_Spark-76B900?logo=nvidia&logoColor=white)
![GB10](https://img.shields.io/badge/GB10-121_GB_unified_memory-5C2D91)

**Vincent Marquez** · 2026

> **Measured result (2026-07-23):** **11.12–11.19 overall decode tok/s** at full top-8, with mid-window peaks up to **~11.7 tok/s** and **AL up to 4.0** — on **one NVIDIA DGX Spark**, **full top-8 (K=8)** routing. No `TOPK=1` shortcut.

A single-machine **GLM-5.2 744B MoE inference** campaign on **NVIDIA DGX Spark** (**GB10 / Grace Blackwell · 121 GB unified memory · aarch64**) using **CUDA fused decode, GPU MLA, prompt lookup decoding (PLD), large expert VRAM tier, managed KV, and CACHE_ROUTE expert residency**.

This repository is the reproducible performance record: benchmark numbers, experimental engine notes, scripts, quality controls, and the upstream **colibrì CACHE_ROUTE PR #199** behind the climb from roughly **2.4 → 3.3 → 9.5 → 11.1+ tok/s**. The ongoing target is **30 tok/s without reducing full top-8 quality**.

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
| Reached **11.1+ overall decode tok/s** (mid peaks **~11.7**, AL=4) running **GLM-5.2 int4** at full **top-8** on one GB10; peak/mid/overall reported separately | [docs/NUMBERS.md](docs/NUMBERS.md) |
| Designed / implemented **CACHE_ROUTE** (opt-in cache-aware MoE routing, default **off**) | Merged upstream: **[PR #199](https://github.com/JustVugg/colibri/pull/199)** |
| Explained the mechanism on the project issue | [#161 comment](https://github.com/JustVugg/colibri/issues/161#issuecomment-4970926845) |
| Wrote a **quality A/B harness** with live progress (stop-safe) | [`scripts/quality-ab-cr.py`](scripts/quality-ab-cr.py) |
| Documented hardware, flags, and lab results | [`docs/`](docs/) |

My **engine code contribution** lives in upstream after merge (commit `62419af` via PR #199).  
This repo keeps **notes, scripts, and results** so others can see the GB10 story in one place.

Fork of the engine (for history / local branches): https://github.com/VincentMarquez/colibri  

---

## Benchmark result — **11.1+ tok/s overall**, full top-8 on DGX Spark

Still on the same box: **NVIDIA DGX Spark · GB10 · 121 GB UMA · aarch64 · local int4**.  
Goal remains **30 decode tok/s at full top-8 quality** (no `TOPK` prune).  
We have **not** hit 30 yet; this is a progress report + open work.

### Headline numbers (timed `./glm`, full K=8, `TEMP=0`) — **2026-07-23**

| cell / tier | mid `[t=16]` | overall decode | AL (tok/fw) | notes |
|-------------|-------------:|---------------:|------------:|-------|
| **k8maxal2** (best overall family) | 10.50–11.35 | **11.09–11.17** | 3.37–3.46 | pure-`7` prompt, WARM=64, NGEN=128, GPU MLA + fuse + ~64–75 GB expert tier |
| **k8idot_union** (amort multi-S) | — | **11.19** | 3.90 | IDOT multi-row binary 2026-07-23 |
| **k8push30** (AL=4 track) | **11.59–11.71** | 10.94–10.98 | **4.00** | full DRAFT=3 PLD accept |
| Prior multi-S + free PLD (historical) | ~9.5 peak | ~8.5–8.6 | up to ~4.0 | superseded by 11.1+ |
| S=1 TOPK=1 (quality reduced) | ~14.7 | ~14.1 | 1.0 | ceiling check only — **not** full-k8 |

**Protocol (honest):** short synthetic context (~40-token prompt of `7` lines), warm discarded, **timed decode only**. Full top-8 (`experts/token ≈ 600 = 8×75`). Speculation is **PLD n-gram** (`FORCE_PLD`), not MTP (0% accept). CACHE_ROUTE J=1 M=32. Expert hit ~**99.9%** once pinned.

**Verbatim best overall (`k8maxal2` → 11.12):**

```text
128 tokens in 11.51s (11.12 tok/s decode) | expert hit rate 99.9% | RSS 11.40 GB | swap 18.6%
experts loaded/token: 600.0 (per-layer 8.00 across 75; baseline topk=8)
speculation: 3.37 tokens/forward | MTP acceptance 0%
CUDA expert tier: 3386 resident experts (64.05 GB)
PROFILE: expert-disk 0.466s | expert-matmul 0.137s | attention 9.275s | lm_head 0.276s | other 1.361s
```

Earlier public points (~2.4 full / ~3.3 CACHE_ROUTE chat / ~9.5 multi-S peak) remain historical rungs.  
Chat vs timed still differ; we keep reporting **decode tok/s** from the timed window after warm.

### What we implemented / measured (local C engine work)

All of this is still **experimental / local** relative to stock upstream unless noted — not claiming a merged PR here:

- **Device-resident S=1 fused decode** (`CUDA_FUSE` + managed KV + GPU MLA): attention/experts stay on the UMA path; PROFILE still folds some expert drain into the attention bucket.
- **Multi-S verify path (S=2..4)** for MTP/PLD: batch MLA + device routers + per-row FusedMoe (or opt-in union). Residual-on-GPU path is required; residual D2H every layer was a multi-S e2e killer.
- **CACHE_ROUTE** still the residency lever (J/M/α cells). Sacred / no-swap modes preserve logits better for accept but thrash more without CR fill.
- **Free drafts first**: PLD / n-gram (`PLD_FIRST` / `FORCE_PLD`) before the MTP head — on list-like prompts this is what drove **tok/fw ≈ 3–4** when it hit.
- **Warm n-gram corpus** after `WARM_RESET_KV` (keep warm tokens for PLD without long KV) — works as plumbing; accept after reset is still weaker than long-context PLD.
- **MoE-Spec-style `EXPERT_BUDGET`**, **MTP_MARGIN** early-stop, **adaptive draft depth**, **MTP_DRAFT_TOPK**, **DRAFT capped at 3** under multi-S fuse (S≤4).
- **Physics takeaway on this silicon:** full-k8 S=1 is roughly a **~7 tok/s** wall (MLA bandwidth + top-8 experts). 30 needs multi-S verify cost **≲ ~1.1× S=1** with sustained **AL ≳ 4**, or a much faster S=1 baseline. Today best net is **~8.5–9.5** when PLD multi-S pays — about **~2.7× short of 30** (11.1 → 30).

### Still working on (toward 30 full-k8)

1. **Faster MLA / attention** — at 11.1+ overall, **~80% of timed decode is attention** (expert disk+matmul near-free once pinned). Pure MLA microbench @ctx128 ~25–26 tok/s.
2. **Cheaper multi-S verify** — amort/union multi-S still does not beat serial PLD much on this box.
3. **Sustained AL ≥ 4 without thrash** — push30 hits AL=4; combine with faster MLA without killing accept.
4. **Honest quality separation** — full top-8 vs CACHE_ROUTE swap% vs TOPK=1 ceiling stay labeled separately (same discipline as #161).

### Goal (unchanged)

> **30 decode tok/s on full top-8 (K=8)** on this GB10 / 121 GB UMA host, without silently collapsing to TOPK=1.

Happy to take protocol notes if the project wants a standard “Spark full-k8 timed” row for the community table. More numbers as the multi-S cost comes down.

— continuing on the local C stack; no claim that 30 is done.

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

**Independent repro (x86 dual 5090):** +42% decode (2.26 → 3.09), ~15% substitution, MMLU-200 **indistinguishable** at n=200 noise — see [docs/QUALITY.md](docs/QUALITY.md). Keep off the main leaderboard table.  

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
