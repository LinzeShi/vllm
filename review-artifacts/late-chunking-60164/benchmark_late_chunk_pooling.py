# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fixed-size chunk means before embedding projection/normalization.

Run from the repository root; use ``--output out.json`` to save samples.
GPU graph timing excludes Python dispatch, compilation and input allocation;
eager wall timing includes dispatch and intermediate/output allocations.
Correctness is checked against independent FP64 per-chunk means before timing.
Input buffers rotate past L2 capacity. Effective bandwidth counts logical
input/output bytes; it does not measure HBM traffic or force output writeback.
"""

import importlib.util
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.pooler.tokwise.heads import _mean_pool_chunks
from vllm.utils.argparse_utils import FlexibleArgumentParser


def split_tail(data: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_size = min(chunk_size, len(data))
    full, tail = divmod(len(data), chunk_size)
    end = full * chunk_size
    result = (
        data[:end]
        .reshape(full, chunk_size, data.shape[-1])
        .mean(1, dtype=torch.float32)
    )
    if tail:
        result = torch.cat(
            (result, data[end:].mean(0, keepdim=True, dtype=torch.float32))
        )
    return result


def direct_output(data: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_size = min(chunk_size, len(data))
    full, tail = divmod(len(data), chunk_size)
    end = full * chunk_size
    result = torch.empty(
        (full + bool(tail), data.shape[-1]), dtype=torch.float32, device=data.device
    )
    torch.mean(
        data[:end].reshape(full, chunk_size, data.shape[-1]),
        dim=1,
        dtype=torch.float32,
        out=result[:full],
    )
    if tail:
        torch.mean(
            data[end:], dim=0, keepdim=True, dtype=torch.float32, out=result[full:]
        )
    return result


def padded(data: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_size = min(chunk_size, len(data))
    tail = len(data) % chunk_size
    pad = chunk_size - tail if tail else 0
    padded_data = F.pad(data, (0, 0, 0, pad)) if pad else data
    result = padded_data.reshape(-1, chunk_size, data.shape[-1]).mean(
        1, dtype=torch.float32
    )
    if tail:
        result[-1].mul_(chunk_size / tail)
    return result


def wall_samples(fn, data, chunk_size, iterations):
    samples = []
    for _ in range(5):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn(data, chunk_size)
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return samples


@torch.inference_mode()
def main(args):
    # FlashInfer's graph fallback uses CUDA events when CUPTI is unavailable.
    from flashinfer.testing import (
        bench_gpu_time_with_cudagraph,
        bench_gpu_time_with_cupti,
    )

    cupti = importlib.util.find_spec("cupti") is not None
    gpu_timer = bench_gpu_time_with_cupti if cupti else bench_gpu_time_with_cudagraph
    torch.manual_seed(0)
    rng = random.Random(args.order_seed)
    implementations = {
        "split_tail": split_tail,
        "direct_output": direct_output,
        "padded": padded,
        "selected": _mean_pool_chunks,
    }
    if args.compiled:
        implementations["compiled_split_tail"] = torch.compile(
            split_tail, fullgraph=True
        )
    cases = [(int(n), int(c)) for n, c in (item.split(":") for item in args.cases)]
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "args": vars(args),
        "gpu_timer": "CUPTI" if cupti else "CUDA graph events (CUPTI unavailable)",
        "cold_l2_cache": True,
        "rows": [],
    }
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for tokens, chunk_size in cases:
            # Multiple documents amortize launch costs; each document is reduced
            # independently, just as in the request-by-request production path.
            data = torch.randn(
                args.batch, tokens, args.hidden_size, device="cuda", dtype=dtype
            )
            reference = [
                torch.stack(
                    [part.double().mean(0) for part in x.split(chunk_size)]
                ).float()
                for x in data
            ]
            order = list(implementations)
            rng.shuffle(order)
            for name in order:
                impl = implementations[name]

                def operation(x, size, impl=impl):
                    return [impl(doc, size) for doc in x.unbind()]

                actual = operation(data, chunk_size)
                max_error = max(
                    (a - r).abs().max().item() for a, r in zip(actual, reference)
                )
                for a, r in zip(actual, reference):
                    torch.testing.assert_close(a, r, atol=2e-6, rtol=2e-5)
                for _ in range(20):
                    operation(data, chunk_size)
                timer_kwargs = dict(
                    input_args=(data, chunk_size),
                    dry_run_iters=5,
                    repeat_iters=25,
                    cold_l2_cache=True,
                )
                if cupti:
                    timer_kwargs["use_cuda_graph"] = True
                gpu_us = [v * 1000 for v in gpu_timer(operation, **timer_kwargs)]
                eager_us = wall_samples(operation, data, chunk_size, args.iterations)
                output_bytes = sum(x.numel() * x.element_size() for x in actual)
                minimum_bytes = data.numel() * data.element_size() + output_bytes
                row = {
                    "implementation": name,
                    "dtype": dtype_name,
                    "tokens": tokens,
                    "chunk_size": chunk_size,
                    "batch": args.batch,
                    "hidden_size": args.hidden_size,
                    "max_abs_error_vs_fp64": max_error,
                    "gpu_us_median": statistics.median(gpu_us),
                    "gpu_us_p10": float(torch.tensor(gpu_us).quantile(0.1)),
                    "gpu_us_p90": float(torch.tensor(gpu_us).quantile(0.9)),
                    "eager_us_median": statistics.median(eager_us),
                    "eager_us_samples": eager_us,
                    "gpu_us_samples": gpu_us,
                    "minimum_bytes": minimum_bytes,
                    "effective_gbps": minimum_bytes / statistics.median(gpu_us) / 1e3,
                }
                result["rows"].append(row)
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if not k.endswith("samples")}
                    ),
                    flush=True,
                )
                Path(args.output).write_text(json.dumps(result, indent=2))
            if args.compiled:
                torch.compiler.reset()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "512:64",
            "513:64",
            "4096:256",
            "4097:256",
            "8192:256",
            "8191:128",
            "128:256",
            "512:1",
        ],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--output", default="late-chunk-pooling.json")
    main(parser.parse_args())
