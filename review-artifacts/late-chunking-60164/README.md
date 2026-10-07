# Late-chunking benchmark reproduction

Supplementary measurements for vLLM PR #60164. These scripts compare reducer
implementations and ordinary, chunked and mixed offline Nomic requests. They are
review artifacts, separate from the vLLM implementation diff.

## Environment and source

- NVIDIA RTX 4070 Laptop GPU, WSL Linux, Python 3.12.
- PyTorch `2.13.0+cu130`, Transformers `5.17.0`, FlashInfer `0.7.0.post1`.
- Compiled vLLM artifacts: `84bcbc62644356270aaaa5e2d0237d03adc9bb3a`.
- Published pooling baseline: `9f7a231117823067d946f05a6aa14aa4963a99aa`.
- Revised pooling source: `f873b469b9661a2f212d14e410834ca568348a84`.
- Nomic model revision: `720244025c1a7e15661a174c63cce63c8218e52b`;
  remote-code revision: `7710840340a098cfb869c4f65e87cf2b1b70caca`.

During the recorded end-to-end comparisons, the three pooling source files
were alternated in the same checkout. The measured revised source matches
`f873b469b9661a2f212d14e410834ca568348a84`. Each JSON
records their SHA256 values. Its `commit` field identifies the surrounding
checkout, not which pooling implementation was temporarily installed. The
source manifest below identifies the old and revised files explicitly.

The final regression suite used Transformers 5.18.0; the performance runs above
used 5.17.0. Earlier A100 validation predates the pooling refactor and is not
presented as validation of this revision.

## Reducer comparison

`benchmark_late_chunk_pooling.py` compares:

- `split_tail`: reduce complete chunks and the tail, then concatenate.
- `direct_output`: allocate the final output and write both reductions into it.
- `padded`: pad to an exact multiple, reduce, and correct the tail divisor.
- `selected`: import the production reducer; exact multiples retain the simple
  reduction, while tails write directly into final output storage.
- Optional `compiled_split_tail`: compile the baseline with `torch.compile`.

The full comparison covers eight length/chunk-size pairs, three input dtypes
and batches of one or eight documents. Every implementation is checked against
independent FP64 chunk means before timing. All 48 conditions passed for the four
main implementations; the maximum absolute error was `5.96e-8`.

GPU timing uses CUDA graph events with input buffers rotating beyond L2 capacity.
CUPTI was unavailable. This excludes Python dispatch and graph-captured
allocations from replay timing. The separate eager measurement includes Python
dispatch and intermediate/output allocations. Effective bandwidth is based on
logical bytes, not measured HBM traffic or forced output writeback.

Representative FP16 GPU medians:

| Documents | Tokens per document | Chunk size | Split and concatenate | Selected |
| --- | --- | --- | --- | --- |
| 1 | 513 | 64 | 7.58 us | 6.14 us |
| 8 | 513 | 64 | 76.39 us | 68.81 us |
| 8 | 4097 | 256 | 238.69 us | 231.53 us |
| 8 | 8191 | 128 | 473.19 us | 464.18 us |

Across the nine tail shape/dtype combinations, the median GPU-time reduction
was 4.2% for batch one and 3.0% for batch eight. Eager timings include regressions
and are not uniformly better. Padding generally costs more in this matrix.
The small compiled comparison did not show a consistent advantage. These
measurements do not establish optimality across hardware or shapes.

## End-to-end comparison

`benchmark_late_chunking.py` runs seven scenarios: ordinary short, ordinary
batch, ordinary long, chunked short, chunked batch, chunked long and mixed.
Inputs have 32, 512, 4096 or 8191 tokens, with model-default compilation enabled.
Each scenario has five warmups and 30 timed repetitions. Startup, compilation
and reference calculation are excluded; tokenization, scheduling and output
copies are included.

Recorded order: revised, old, old, revised, then revised with async scheduling.
All five runs passed their output checks against native token outputs. Minimum
cosine similarity was `0.999957`. This checks chunk-reduction behavior, not
independent Transformers model parity or retrieval quality.

For the 8191-token chunked case, old/new medians were 162.2/152.1 ms in the first
pair and 152.3/150.7 ms in the second. Between-run variation is substantial, so
these results do not establish a robust end-to-end speedup.

## Reproduction

Use an installed vLLM CUDA development environment with the versions above.
Keep these scripts outside the implementation checkout. Run from the vLLM
repository root, setting `benchmark_artifacts` to this directory:

```bash
benchmark_artifacts=/path/to/late-chunking-benchmarks
OMP_NUM_THREADS=4 PYTHONPATH="$PWD" .venv/bin/python \
  "$benchmark_artifacts/benchmark_late_chunk_pooling.py" \
  --batch 1 --output reducer-batch1.json
OMP_NUM_THREADS=4 PYTHONPATH="$PWD" .venv/bin/python \
  "$benchmark_artifacts/benchmark_late_chunk_pooling.py" \
  --batch 8 --output reducer-batch8.json
OMP_NUM_THREADS=4 PYTHONPATH="$PWD" .venv/bin/python \
  "$benchmark_artifacts/benchmark_late_chunk_pooling.py" \
  --compiled --dtypes float16 float32 --cases 513:64 8191:128 \
  --output reducer-compiled.json
OMP_NUM_THREADS=4 PYTHONPATH="$PWD" .venv/bin/python \
  "$benchmark_artifacts/benchmark_late_chunking.py" \
  --iterations 30 --output end-to-end.json
```

The reducer script imports the new helper, so run it on the revised source.
Run the end-to-end script on both comparison revisions, restarting the process
between them. Add `--async-scheduling` for the async case. Optional `HF_HOME`
selects the model cache; `HF_HUB_OFFLINE=1` requires the pinned files to be cached.
Avoid other GPU workloads or concurrent test runs while measuring performance.

## Measurements and provenance

- [Result files](results/): all per-case samples, summaries and the compiled
  comparison, including slower and noisy cases.
- [Source manifest](source-manifest.json): production-file and script hashes.
- [Result provenance](result-provenance.json): original and shared JSON hashes.
  Only absolute output paths were shortened to filenames; measurements are
  unchanged.

