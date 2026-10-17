# Bounded target-token prefix reuse: TP=4 measurements

Qwen3-32B on 4×A100 80GB SXM, TP=4, ordinary BF16, FlashAttention 2,
torch.compile/CUDA graphs, synchronous scheduling. The table measures offline
teacher scoring, including one generated token per response.

## Recorded results

`performance-table.md` contains all eight cases. Each case starts cold and
alternates six repetitions per path. Each table row sums four group medians,
covering 16 responses. `summary.json` also records per-repetition totals.

`cloud/repro/` contains the output arrays and raw timing records. To verify the
matrices and regenerate the table without a GPU, run this from the bundle root
using an environment with NumPy:

```bash
/path/to/vllm/.venv/bin/python analyze.py
```

The fixed input file is `repro/generated.json`, SHA256
`5832d8354af07ba3ab2998354a73f1ca466312b0667140a6024641d4578aa906`.
It contains eight prompt groups and four student responses per group, generated
with Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`.
The prompt groups come from `zhuzilin/dapo-math-17k` and the Qasper subset of
`THUDM/LongBench`; source rows and complete token inputs are included. Some math
responses reached the 1024-token generation cap.

## Run the measured revision

The recorded run used vLLM source commit
`90d6d871d6700dd229a76b788bf0dca22bab6c70` plus `measured.patch`, whose SHA256 is
`f2586c3a2fb761cdc27bec8f2a14553645f64dc3034f86acbe1c44f409c1a4ee`.
The PR subsequently added an actual-KV-format admission check; the recorded
BF16 path remains eligible, and score computation is unchanged.

Use an editable checkout at that base with compatible CUDA extensions.
`environment.json` records the source/wheel commits and runtime
versions. The run used PyTorch 2.14.0+cu130 and a precompiled vLLM wheel from
`39983a5e7a5a575a7829f5d5ada50a33d9337ec1`; the compiled-source compatibility and
runtime source hashes were checked separately from the Python patch.

From the clean, installed checkout, with `opd_repro` pointing to this bundle:

```bash
opd_repro=/absolute/path/to/this/bundle
git apply "$opd_repro/measured.patch"
export PATH="$PWD/.venv/bin:$PATH"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
unset VLLM_BATCH_INVARIANT VLLM_GPU_SYNC_CHECK NCCL_P2P_DISABLE
for budget in 512 2048; do
  .venv/bin/python "$opd_repro/repro/probe.py" \
    --phase evaluate --tp 4 --budget "$budget" \
    --label "rerun-tp4-b${budget}" --performance
done
```

The probe pins the teacher revision to
`9216db5781bf21249d130ec9da846c4624c16137`. It keeps normal TP communication,
sets 2048 KV blocks of 16 tokens, limits each batch to four requests, and uses
the same saved response IDs for both paths. Read `protocol.json` for the complete
measurement conditions. Output goes to new `repro/rerun-*` directories; the
recorded results are preserved.

## Interpretation

Staggered arrivals target 20 ms gaps but are admitted at engine-step boundaries.
The measurements do not represent HTTP traffic or complete training throughput.
BF16 results are not bit-exact; these records establish score completeness and
report numerical differences, not training-quality equivalence or a universal
error bound. They also do not compare TP scaling or different GPU generations.

## Additional async-preemption validation

`additional-validation/` contains a separate Qwen3-0.6B synthetic test and its
arrays, run against the current PR code. It compares uncached scoring, bounded
reuse, and bounded reuse with forced prefill preemption. Each path scores eight
200-token responses with an actual token, an extra candidate, and padding per
row. Both cached paths were bit-exact with each other. Against uncached scoring,
ordinary BF16 reached maximum logprob differences of 0.142422 on actual tokens
and 1.171719 on extra candidates; batch-invariant mode matched all three paths
exactly. This synthetic test is separate from the 32B timing workload.

To repeat it on the current PR checkout with one GPU, use new output labels:

```bash
VLLM_GPU_SYNC_CHECK=warn VLLM_BATCH_INVARIANT=0 \
  .venv/bin/python "$opd_repro/additional-validation/probe_async_pressure.py" rerun-bf16
VLLM_GPU_SYNC_CHECK=warn VLLM_BATCH_INVARIANT=1 \
  .venv/bin/python "$opd_repro/additional-validation/probe_async_pressure.py" rerun-bi
```

These runs used synchronization warning mode on WSL and eager execution. They
are correctness/numerical probes, not latency measurements or training checks.
