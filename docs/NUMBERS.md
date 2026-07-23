# GLM-5.2 on NVIDIA DGX Spark — **11.1+ tok/s overall** (Full Top-8)

**Metric:** **decode tok/s** from the engine stats line (generation window only).  
**Not** overall wall tok/s (includes prefill / turn overhead).  
**Width:** full **K=8** unless noted.  
**Host:** one NVIDIA DGX Spark · GB10 / Grace Blackwell · 121 GB unified memory · aarch64 · disk **9.69 GB/s** O_DIRECT.

## Current best — 2026-07-23 (full top-8, short-context timed decode)

| cell | mid `[t=16]` | overall decode | AL (tok/fw) | expert hit | when |
|------|-------------:|---------------:|------------:|-----------:|------|
| **k8maxal2** | 10.50 | **11.12** | 3.37 | 99.9% | 07:23 UTC |
| k8w16_maxal_seed | 11.27 | **11.17** | 3.46 | ~100% | 12:55 UTC (wave16) |
| k8w16_maxal_d4 | 11.35 | 11.16 | 3.46 | ~100% | 12:56 UTC |
| k8idot_union | — | **11.19** | 3.90 | 99.9% | 09:12 UTC |
| k8push30 | **11.71** | 10.94–10.98 | **4.00** | ~100% | mid-morning / wave16 |
| t1hot (TOPK=1, **not full-k8**) | 14.69 | 14.06 | 1.0 | 100% | diagnostic only |

### Protocol for the 11.1+ numbers

- Prompt: ~40 tokens synthetic (`7` lines × 20), `CHAT_TEMPLATE=0`
- Warm: 48–64 tokens discarded, then timed NGEN=128–160
- Routing: full top-8 (`experts loaded/token ≈ 600`); `CACHE_ROUTE=1 J=1 M=32`
- Spec: `FORCE_PLD=1` n-gram drafts (`DRAFT=3`), **MTP off**
- CUDA: unified + dense + grouped + **GPU MLA** + **CUDA_FUSE** + `CUDA_EXPERT_GB≈70–85` (~64 GB device experts after warm pin)
- Expert hit after warm pin: **~99.9%**

### Verbatim 11.12 overall

```text
128 tokens in 11.51s (11.12 tok/s decode) | expert hit rate 99.9% | RSS 11.40 GB | swap 18.6%
experts loaded/token: 600.0 (per-layer 8.00 across 75; baseline topk=8)
speculation: 3.37 tokens/forward | MTP acceptance 0%
CUDA expert tier: 3386 resident experts (64.05 GB)
PROFILE: expert-disk 0.466s | expert-matmul 0.137s | attention 9.275s | lm_head 0.276s | other 1.361s
```

### Physics

| bucket (128-tok timed window @ 11.12) | time | share |
|--------------------------------------|-----:|------:|
| attention (GPU MLA; includes some fuse drain) | 9.275s | ~81% |
| expert-disk | 0.466s | ~4% |
| expert-matmul | 0.137s | ~1% |
| lm_head + other | ~1.6s | ~14% |

**Bottleneck is MLA**, not experts/disk, once the VRAM expert tier is hot.

---

## Wave16 exclusive re-measure (2026-07-23 12:50–13:03 UTC)

| cell | mid16 | overall | AL |
|------|------:|--------:|---:|
| k8maxal2 | 10.53 | 11.09 | 3.37 |
| k8push30 | 11.59 | 10.98 | 4.00 |
| k8w16_maxal_seed | 11.27 | **11.17** | 3.46 |
| k8w16_maxal_d4 | 11.35 | 11.16 | 3.46 |
| k8w16_push_idot | 10.51 | 10.22 | 4.00 |
| k8w16_amort | 11.68 | 10.60 | 3.90 |
| k8w16_serial_hot | 11.61 | 10.51 | 3.90 |
| k8w16_maxal_gb85 | 10.71 | 10.61 | 3.46 |

No WIN30. Ceiling remains **~11.1–11.2 overall** on this stack.

---

## Historical progression

### Tier A — stock routing (CACHE_ROUTE off)
- Timed PROFILE: **2.39** tok/s · hit ~82%
- Chat warm: **~2.08** tok/s

### Tier B — CACHE_ROUTE early
- Chat ladder: **3.33** tok/s · hit 97% · swap 14%

### Tier C — fused S=1 + device-tier
- CR M16 + device-tier: **5.63–6.17** tok/s
- Multi-S + free PLD historical: mid ~9.5 / overall ~8.5–8.6 (superseded)

### Tier D — current (PLD + GPU MLA + large expert pin)
- **11.1+ overall full top-8** (see table above)

---

## Goal

> **30 decode tok/s on full top-8 (K=8)** without collapsing to TOPK=1.

Status: **not reached** (~2.7× gap from 11.1). Next lever: MLA speed.
