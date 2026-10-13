# Online late chunking measurements

These results compare server chunking with exporting unnormalized token vectors and computing the same chunk means, normalization and source ranges on the client. Both use the same multilingual E5 model and token boundaries. This is an HTTP payload and client-observed latency comparison, not a GPU kernel speedup claim.

Environment: NVIDIA A100 SXM 80GB, BF16, Python 3.12.3, PyTorch 2.13.0+cu130, Transformers 5.17.0. Model revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Source base: `e6e84f350` with the CDE changes. Full runtime versions and raw samples are included here. Validated feature source: `9f99169b7de8544d664d65fee9237c283e5adeec`; the final change only removes duplicate test cases.

## Results

Client and server run on the same machine over loopback HTTP. Each case has two warmups and nine measured requests per mode, with alternating order. Latency includes JSON parsing and the client's offset generation and pooling. The response byte count includes JSON metadata. p95 is an empirical percentile over nine samples.

| Tokens/document | Batch | Chunk size | JSON bytes: client / server | Median ms: client / server | p95 ms: client / server |
| --- | --- | --- | --- | --- | --- |
| 52 | 1 | 16 | 252,439 / 32,548 | 47.17 / 20.87 | 54.44 / 21.98 |
| 244 | 1 | 32 | 1,175,606 / 64,783 | 114.59 / 23.86 | 119.37 / 26.43 |
| 436 | 4 | 64 | 8,384,058 / 226,011 | 481.07 / 52.75 | 487.08 / 56.10 |
| 52 | 1 | 1 | 252,439 / 419,670 | 50.58 / 61.54 | 51.15 / 62.38 |

All paired source ranges agreed; the minimum vector cosine was 0.99999839. The chunk size 1 case increased response size and latency. Worker CUDA allocator peaks were identical for the paired cases; these measurements do not establish lower GPU or host memory usage. This is a small serving experiment, not a retrieval quality evaluation or a production throughput benchmark.

## Reproduce

Use the CDE source and an environment with its test dependencies installed. Start the service:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm serve intfloat/multilingual-e5-small --runner pooling \
  --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --dtype bfloat16 --enforce-eager --max-model-len 512 \
  --pooler-config.task token_embed --gpu-memory-utilization 0.5
```

Then measure the HTTP path:

```bash
python measure-cde.py online --url http://localhost:8000 --runs 9 \
  --output http-measurements.json
```

Stop that service before running the independent offline memory measurement. On this disposable local test engine, enable serialization for the measurement callbacks:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 python measure-cde.py memory \
  --output memory-measurements.json
```

The memory results record three warmed samples per mode. Allocated and reserved CUDA memory are reported separately. No benchmark scripts are added to the upstream source patch.
