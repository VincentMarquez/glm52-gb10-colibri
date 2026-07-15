# Measured numbers (honest tiers)

**Metric:** **decode tok/s** from the engine stats line (generation window only).  
**Not** overall wall tok/s (includes prefill / turn overhead).  
**Width:** full **K=8** unless noted.  
**Host:** NVIDIA DGX Spark · GB10 · 121 GB unified · disk **9.69 GB/s** O_DIRECT.

---

## Tier A — stock routing (CACHE_ROUTE off)

Fair full‑k8 baseline (leaderboard-comparable *routing*).

| Cell | decode tok/s | hit | notes |
|------|-------------:|----:|-------|
| Timed PROFILE (WARM=160, NGEN=48, greedy) | **2.39** | ~82% | non-interactive |
| Chat warm (primes, `:reset` each turn) | **~2.08** | ~82% | multi-turn chat |

Context: public community full-quality peaks on other machines were often **~0.4–2.06** decode (e.g. Metal M5 Max ~2.06 in older README discussion). Different hardware — not apples-to-apples, but useful backdrop.

---

## Tier B — experimental CACHE_ROUTE (early stack)

**Config sketch:** CUDA unified + dense/grouped experts + warm session LRU.  
**Before** fused-device decode (D4) / heavy device-tier.

`CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=12`

| Cell | decode tok/s | hit | swap | notes |
|------|-------------:|----:|-----:|-------|
| Chat ladder best | **3.33** | **97%** | **14%** | multi-turn primes + `:reset` |
| Timed one-shot | **2.67** | 93.5% | 15% | less favorable than multi-turn chat |

**Routing-side jump people care about:** ~**2.4 → 3.33**.

**Independent x86 repro** (dual 5090 / Gen5, PCIe, not unified memory): true top-8 **2.255** → J2 M12 **3.09** (+42%), hit 92%, swap **15.3%** — matches the GB10 delta and swap rate. Quality: MMLU-200 59% vs 62% (noise; **no detectable difference**). See [QUALITY.md](QUALITY.md). **Not** a main-table row.

Example footer (Tier B peak):

```text
41 tok · 3.33 decode tok/s · hit 97% · swap 14% · RSS 78.5 GB
I/O ~1.5s · expert ~3.2s · attention ~4.1s
```

---

## Tier C — later stack on the same GB10

Same host, still full K=8. On top of CACHE_ROUTE: local **GPU MLA (D3)**, **fused S=1 decode (D4 / CUDA_FUSE)**, optional **device-tier** hot experts (`CUDA_EXPERT_GB`), **PILOT**, often **M=16**, sometimes **ROUTE_ALPHA=0.5**.

### These are **not** “CACHE_ROUTE alone”

| Cell | CR | decode tok/s | hit | swap / agree | notes |
|------|-----|-------------:|----:|--------------|-------|
| Strict, later stack, CR off | off | **2.98** | 86.2% | — | fair A/B partner |
| CR M16, same later stack | J2 M16 | **5.14** | 94.0% | swap 19.6% · agree 80.4% | **only CR differs** from row above |
| CR M16, managed experts | J2 M16 | **5.27** | 96.3% | — | fuse+MLA+pilot |
| CR M16 + device-tier ~50 GB | J2 M16 | **5.63–5.64** | ~96% | ~19% / ~81% | mid-window higher |
| CR M16 + device-tier (confirm) | J2 M16 | **6.17** | 96.9% | 18.4% / 81.6% | full 48-tok timed window |

### Takeaways

| Claim | Support |
|-------|---------|
| CR alone (early) | **~2.4 → 3.33** chat |
| CR on/off with later stack fixed | **2.98 → 5.14** |
| Full later stack + CR peak | **~5.6–6.2** decode |
| Strict full-k8 without CR | still ~**3** class even with fuse |

---

## What we did **not** claim as full-k8 leaderboard

| Trap | Why it’s wrong |
|------|----------------|
| **TOPK=1** ~4.22 tok/s as “full quality” | Pruned routing — quality trade |
| **6 tok/s = CACHE_ROUTE alone** | Needs Tier C CUDA/residency stack |
| **Overall** wall tok/s | Includes prefill; not the ranking metric |
| **Default CR** | Experimental; quality gates still limited |

---

## iobench (disk)

| Mode | GB/s |
|------|-----:|
| Buffered | 4.25 |
| O_DIRECT | **9.69** |

---

## Reproduction sketch

Exact local scripts vary; conceptual cells:

```bash
# Tier A — stock full-k8
unset CACHE_ROUTE TOPK
# ... COLI_CUDA / unified / PIN / warm as on your box ...
./coli chat   # or timed glm PROFILE / WARM+NGEN cell

# Tier B — experimental CR
CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=12 ./coli chat

# Tier C — machine-specific; see local run-glm52-k8*.sh
# Report every flag if you publish a number.
```

Always report: **decode tok/s**, **hit%**, **swap%** if CR on, **K width**, **MTP on/off**, **context / warm protocol**.
