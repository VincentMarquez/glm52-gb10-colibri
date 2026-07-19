# My GLM-5.2 work on 1 DGX Spark (GB10)

**Vincent Marquez** · 2026  

> **9.5 tok/s peak** on one NVIDIA DGX Spark at full top-8 (K=8), with **~8.5–8.6 tok/s overall decode** on the best run.
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

## Update — DGX Spark (GB10) full top-8 campaign toward **30 tok/s**

Still on the same box: **NVIDIA DGX Spark · GB10 · 121 GB UMA · aarch64 · local int4 + int8 MTP**.  
Goal for this thread remains **30 decode tok/s at full top-8 quality** (no `TOPK` prune).  
We have **not** hit 30 yet; this is a progress report + open work, not a leaderboard claim of 30.

### Headline numbers (timed `./glm`, full K=8, `TEMP=0`)

| tier | mid `[t=16]` | overall decode | tok/fw | notes |
|------|--------------|----------------|--------|-------|
| **S=1 fuse, full k8 + CR** | **~6.7–7.3** | **~6.5–6.9** | 1.0 | hot experts (~99% hit), short-context warm |
| **Multi-S + free PLD + CR** | **~9.5 peak** | **~8.5–8.6 peak** | up to **~4.0** | best full-k8 e2e so far (high variance) |
| S=1 TOPK=1 (quality reduced) | ~14 | ~13 | 1.0 | ceiling check only — **not** full-k8 |

Earlier public points on this issue (~2.4 full / ~3.3 CACHE_ROUTE chat) are superseded on the **timed decode** path by the fused S=1 + residency stack.  
Chat vs timed still differ; we keep reporting **decode tok/s** from the timed window after warm.

### What we implemented / measured (local C engine work)

All of this is still **experimental / local** relative to stock upstream unless noted — not claiming a merged PR here:

- **Device-resident S=1 fused decode** (`CUDA_FUSE` + managed KV + GPU MLA): attention/experts stay on the UMA path; PROFILE still folds some expert drain into the attention bucket.
- **Multi-S verify path (S=2..4)** for MTP/PLD: batch MLA + device routers + per-row FusedMoe (or opt-in union). Residual-on-GPU path is required; residual D2H every layer was a multi-S e2e killer.
- **CACHE_ROUTE** still the residency lever (J/M/α cells). Sacred / no-swap modes preserve logits better for accept but thrash more without CR fill.
- **Free drafts first**: PLD / n-gram (`PLD_FIRST` / `FORCE_PLD`) before the MTP head — on list-like prompts this is what drove **tok/fw ≈ 3–4** when it hit.
- **Warm n-gram corpus** after `WARM_RESET_KV` (keep warm tokens for PLD without long KV) — works as plumbing; accept after reset is still weaker than long-context PLD.
- **MoE-Spec-style `EXPERT_BUDGET`**, **MTP_MARGIN** early-stop, **adaptive draft depth**, **MTP_DRAFT_TOPK**, **DRAFT capped at 3** under multi-S fuse (S≤4).
- **Physics takeaway on this silicon:** full-k8 S=1 is roughly a **~7 tok/s** wall (MLA bandwidth + top-8 experts). 30 needs multi-S verify cost **≲ ~1.1× S=1** with sustained **AL ≳ 4**, or a much faster S=1 baseline. Today best net is **~8.5–9.5** when PLD multi-S pays — about **~3× short of 30**.

### Still working on (toward 30 full-k8)

1. **Cheaper multi-S verify** — expert path still too heavy vs S=1 (shared CUDA expert scratch forces serial S positions; true multi-stream needs per-stream scratch).
2. **Sustained high AL without long-context tax** — long warm KV helps PLD (AL up to ~4) but inflates attention; short-ctx + corpus is not yet a net win.
3. **MTP accept vs draft cost** — full-k8 MTP can accept well but draft “other” can erase the win; free PLD is the reliable multi-S fuel so far.
4. **Honest quality separation** — full top-8 vs CACHE_ROUTE swap% vs TOPK=1 ceiling will stay labeled separately (same discipline as the original #161 two-tier report).

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
