# Hardware

## Box

| Field | Value |
|-------|--------|
| Product | NVIDIA **DGX Spark** |
| Codename / node | `spark-79fb` (example hostname) |
| Platform id | `NVIDIA_DGX_Spark` |
| CPU | 10× Cortex-**X925** @ ~3.9 GHz + 10× Cortex-**A725** @ ~2.8 GHz (**20** cores) |
| GPU | NVIDIA **GB10** (sm_121) |
| Memory | **121 GB** unified (CPU+GPU shared) |
| Driver (as measured) | 580.126.09 |
| OS | Ubuntu **24.04** · Linux **6.17** · **aarch64** |
| Storage | Local NVMe (model shards on disk) |

## Disk bandwidth (engine-style random reads)

`c/iobench` · 19 MB × 64 reads · 8 threads · shard e.g. `out-00099.safetensors`:

| Mode | GB/s | Ratio |
|------|-----:|------:|
| Buffered (page cache) | **4.25** | 1.0× |
| **`DIRECT=1` / O_DIRECT** | **9.69** | **2.3×** |

### Flag this louder: `DIRECT=1` on real NVMe

On this DGX Spark local NVMe, **O_DIRECT is a striking ~2.3×** over buffered reads. That matches a page-cache **double-buffering tax** on a fast drive: buffered I/O copies through the page cache; O_DIRECT feeds the engine closer to wire speed.

```bash
# Fair “true disk” number for community tables
./iobench /path/to/shard.safetensors 19 64 8 1   # last 1 = O_DIRECT

# Runtime (native Linux + real NVMe): prefer
DIRECT=1 ./coli …
```

**Per-tier lever (not universal):**

| Storage | What we see | Guidance |
|---------|-------------|---------|
| **Fast local NVMe** (this box) | 4.25 → **9.69 GB/s** | **`DIRECT=1` worth defaulting** for decode |
| Slow / VHDX-backed WSL disks | often ~0.2–1 GB/s class, disk-bound first | O_DIRECT may not move the needle; measure both |

Always publish **both** buffered and O_DIRECT iobench numbers so rows stay comparable.

## Model (as measured)

| Field | Value |
|-------|--------|
| Architecture | GLM-5.2 MoE + DSA family (colibrì snap) |
| Quant | **int4** experts (streaming) · dense resident |
| MTP | int8 MTP heads available; many speed cells used **MTP off** |
| Width | **full top‑8** experts/token unless noted (`TOPK` unset) |

Typical working set on this host with large pin/LRU: on the order of **~75–95 GB RSS** depending on pin budget and warm state.

## Why this box matters for MoE

Streaming MoE is often **disk-bound**. High O_DIRECT bandwidth + large unified memory (big pin ∪ LRU) moves the bottleneck toward **attention + expert matmul** once hit rate is high — which is where CACHE_ROUTE and CUDA path work show up.
