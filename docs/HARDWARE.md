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

| Mode | GB/s |
|------|-----:|
| Buffered | **4.25** |
| **O_DIRECT** | **9.69** |

Community rows often list both. O_DIRECT is the fairer “true disk” number when page cache is huge.

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
