# Finding: cross-node CUDA graphs work on the AEON v0.27.1 sm_121a image, at TP=4 + DCP=4, on four DGX Sparks

Date of measurement: 2026-08-17. Fleet: 4x NVIDIA DGX Spark (GB10, sm_121a, 121 GB unified memory each),
two 200G RoCEv2 rails between nodes (MTU 9000), NCCL 2.30.4.

## The claim being tested

AEON `2026-08-16-v0.27.1` release notes: "#48053 thread_local capture extended to all 5 graph sites (2x-Spark
TP=2; cross-node graphs still broken upstream, bring up TP=2 --enforce-eager)".

## What ran

- Image: `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-08-16-v0.27.1` (vLLM `v0.27.1+aeon.sm121a.dspark`, in
  `site-packages`) + `local-inference-lab/b12x@334a2d75` (cutlass-dsl 4.6.0, matching the AEON pin) + the 30
  overlays in `overlays/` (baked, no runtime mounts). Built with `Dockerfile` here.
- Model: GLM-5.2, unpruned 744B, `QuantTrio/GLM-5.2-Int4-Int8Mix`, MTP k=2 drafter from the same checkpoint.
- Parallelism: TP=4 across the four nodes, DCP=4 (`--dcp-comm-backend a2a`), one GPU per node.
- Attention: `B12X_MLA_SPARSE` (b12x sparse MLA + indexer on GB10), KV cache `nvfp4_ds_mla`.
- Graphs: `--compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[3,6,9,12]}'`.
- Context 32768, `--max-num-seqs 4`, `--max-num-batched-tokens 2048`, `--gpu-memory-utilization 0.90`.
- Exact launcher: `launch/serve_glm52_tp4_dcp4.sh`.

## Evidence

1. Boot: first try, ~8 minutes to KV allocation. KV pool 1,075,968 tokens at 32K.
2. Graph capture: the workers captured PIECEWISE graphs for the mixed prefill/decode sizes and FULL graphs
   for the uniform decode sizes on all four ranks, cross-node. (Verbatim capture log lines from all four
   ranks are appended below once the fleet is back up; the numbers in this file are from the run itself.)
3. Correctness gate: pass on all probes, including 4 and 8 concurrent requests reading real words, and the
   arithmetic probe (17*23 -> 391 at temperature 0).
4. Throughput, same probes as the fleet's production stack (prompt tokens / concurrency; e2e tok/s,
   per-request decode tok/s, prefill tok/s):

| shape | AEON-base platform | stock v0.27.0 fleet stack (same overlays, runtime mounts) |
|---|---|---|
| 1K prompt, C4 | 35.80 e2e / 10.80 per-req / 529 prefill | 35.93 / 11.19 / 566 |
| 14K prompt, C4 | 14.04 e2e / 574 prefill | 13.85 / 559 |

Parity within noise. The graphs are captured, used at decode, and the numbers match the stock stack that
also captures graphs cross-node.

## The flags that matter for a 4-node boot (each one cost us a failed boot before we learned it)

- `--disable-custom-all-reduce`: without it a 300 s MNNVL/TCPStore stall at init on this fleet.
- `VLLM_ONE_GPU_PER_NODE=1`: skips the intra-node gloo mesh race with one GPU per node.
- `--cpu-distributed-timeout-seconds 1800`: the head node loads weights last and slowest; workers with the
  600 s default time out waiting for it and the boot dies with a gloo "connection closed by peer".
- Launch workers first (reverse rank), head last: gloo `connectFullMesh` retries only three times.
- Drop the page cache on every node before launch: unified memory counts cached file pages against vLLM's
  free-memory check; a boot right after a large copy fails on memory it does not really lack.
- `--entrypoint vllm ... serve`: the AEON image entrypoint is `/bin/bash`.
- Persistent kernel caches (`B12X_CUTE_COMPILE_CACHE_DIR`, `CUTE_DSL_CACHE_DIR`, `TRITON_CACHE_DIR`) on the
  weights volume: the b12x CuTe kernels JIT for many minutes otherwise.

## What we did NOT do

- No patch to the AEON tree. Six overlay targets are files AEON also modified in v0.27.1
  (`vllm/config/speculative.py`, `vllm/utils/torch_utils.py`, `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/kv_cache_interface.py`, `vllm/v1/worker/gpu/cudagraph_utils.py`,
  `vllm/v1/worker/gpu/spec_decode/speculator.py`). Two (`torch_utils.py`, `kv_cache_interface.py`) are
  3-way merged (base stock v0.27.0, ours, AEON; `git merge-file` clean, deltas do not overlap; the merge
  keeps AEON's `nvfp4_kv_cache_split_views` for the Triton NVFP4-KV path). The other four keep the AEON
  versions untouched.
- We did not test TP=2 on two Sparks; this is a four-node result.

## Reproduce

```bash
docker build -t glm52-spark-platform:aeon-0.27.1 .
NODE_IPS="..." WEIGHTS_DIR=... launch/serve_glm52_tp4_dcp4.sh
# then any OpenAI-compatible client at :8210; watch docker logs for the capture progress bars
```
