# Bounded prefix reuse scoring experiment

Scripts and results for the target-token scoring RFC. The prototype caps a local prefix-cache hit at `prompt_logprob_start`. It does not implement production request admission, topology checks, or frontend integration.

## Recorded environment

- vLLM commit `97bd8c74ebf12c3d847fb59f24966a83bf6f6920`, wheel `0.30.1rc1.dev695+g97bd8c74e`.
- Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Python 3.12, PyTorch `2.13.0+cu130`, Transformers `5.18.0`, NVIDIA driver 580.159.04, RTX 4090 GPUs.
- BF16, Model Runner V2, eager execution, synchronous scheduling, 16-token blocks, 2048 KV blocks, at most four requests.
- TP=1 with prefill budgets 512 and 2048; TP=2/4 with budget 512. Each configuration has ordinary BF16 and batch-invariant numerical controls. TP=2/4 batch-invariant runs are numerical checks only.

## Inputs

Four DAPO math prompts and four LongBench Qasper document questions, with four model-generated responses per prompt. `inputs.json` records dataset rows, prompt-token hashes, and the exact generated response token IDs. `prepare_inputs.py` retrieves source prompts and verifies the hashes before writing the replay input.

- [DAPO](https://huggingface.co/datasets/zhuzilin/dapo-math-17k): rows 0–3 from the dataset viewer. The viewer snapshot was not revision-pinned; the recorded token hashes reject changed content.
- [LongBench](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672): Qasper rows 0, 1, 3, 8, the first four complete rendered prompts between 1024 and 4096 tokens. Source contexts are downloaded from the dataset, not included here.
- Generation: thinking disabled, temperature 0.8, top-p 0.95, seeds 3100–3103, maximum 1024 tokens. Ten of sixteen math responses reached that limit; no Qasper response did. Recorded responses are replayed unchanged.

## Replay

Use a Linux CUDA 13 environment with enough GPUs for the chosen TP setting. From this directory:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python \
  'https://wheels.vllm.ai/97bd8c74ebf12c3d847fb59f24966a83bf6f6920/vllm-0.30.1rc1.dev695%2Bg97bd8c74e-cp38-abi3-manylinux_2_28_x86_64.whl' \
  'transformers==5.18.0'
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='model')
PY
.venv/bin/python prepare_inputs.py
bash replay.sh
```

`replay.sh` runs the recorded eight configurations and an independent Transformers full-forward reference. Output directories must not already exist. To run only one single-GPU configuration:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
.venv/bin/python probe.py --phase evaluate --tp 1 --budget 512 \
  --label v2-tp1-b512-bi1 --performance
```

To generate a new response set, remove `responses` from each group in `generated.json`, save the resulting document as `prompts.json`, and run `probe.py --phase generate --label generation-new --response-cap 1024` with the same runner environment. This replaces `generated.json`; use the stored response IDs to reproduce the published comparison instead of regenerating them.

## Recorded results

To check the saved numerical results without a GPU:

```bash
tar -xzf results.tar.gz
uv run --no-project --with numpy python check_results.py
```

`summary.json` contains timings and numerical comparisons. `results.tar.gz` contains the result records, exit statuses, and score arrays. Input-token arrays were omitted from the archive; prompts are reconstructed by the hash-checked preparation script. Score arrays are unchanged.

Each timing group starts with an empty cache. Six alternating A/B repetitions use identical tokens. The reported ratio is the sum of per-prompt median baseline times divided by the corresponding sum for bounded reuse. Staggered arrivals target 20 ms gaps and are admitted at synchronous engine-step boundaries. Timing includes one generated token and cache creation, and excludes student generation and learner training.

TP=1 batch-invariant speedups:

| Workload | Prefill budget | Together | Staggered |
| --- | --- | --- | --- |
| Math | 512 | 1.12x | 1.12x |
| Math | 2048 | 1.09x | 1.01x |
| Qasper | 512 | 3.82x | 3.82x |
| Qasper | 2048 | 3.12x | 1.80x |

The 155 saved batch-invariant score comparisons match elementwise. Ordinary BF16 and cross-engine differences are retained in the summary. These small-model results do not measure the default VIME teacher deployment or full training throughput.

The first probe version used an internal request ID to look up externally named outputs in staggered replay and failed with `KeyError`. `probe-v1.py` and `attempt-v1` records preserve those failed attempts. The corrected eight runs all completed. Earlier generation used a 256-token limit and capped all math responses; final measurements use the 1024-token response set described above.
