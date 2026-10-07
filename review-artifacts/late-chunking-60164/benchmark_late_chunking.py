# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure ordinary, chunked and mixed offline Nomic requests end to end.

Run the same command on the comparison revisions. Startup, compilation and
reference calculation are excluded; tokenization, scheduling and output copies
are included. This checks chunk reduction against native token outputs, not HF
model parity.
"""

import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm import LLM, PoolingParams
from vllm.config import PoolerConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser


def main(args):
    llm = LLM(
        "nomic-ai/nomic-embed-text-v1",
        revision="720244025c1a7e15661a174c63cce63c8218e52b",
        code_revision="7710840340a098cfb869c4f65e87cf2b1b70caca",
        trust_remote_code=True,
        runner="pooling",
        pooler_config=PoolerConfig(task="token_embed"),
        dtype=args.dtype,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=args.async_scheduling,
    )
    tokenizer = llm.get_tokenizer()
    paragraph = (
        "The city opened a new library beside the river. Residents borrow books "
        "and use the reading rooms. The railway connects the city with nearby "
        "farms, where researchers measure soil moisture and study irrigation. "
    )
    ids = tokenizer.encode(
        "search_document: " + paragraph * 300, add_special_tokens=False
    )
    texts = {}
    for length in [32, 512, 4096, 8191]:
        text = tokenizer.decode(ids[: length - 2], clean_up_tokenization_spaces=False)
        assert len(tokenizer.encode(text)) == length
        texts[length] = text
    cases = [
        ("ordinary_short", [32], [None]),
        ("ordinary_batch", [32] * 8, [None] * 8),
        ("ordinary_long", [8191], [None]),
        ("chunked_short", [512], [64]),
        ("chunked_batch", [512] * 8, [64] * 8),
        ("chunked_long", [8191], [256]),
        ("mixed", [32, 512, 4096, 8191], [None, 64, None, 256]),
    ]
    result = {
        "args": vars(args),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "platform": platform.platform(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("vllm/model_executor/layers/pooler/tokwise") / f
                for f in ["heads.py", "methods.py", "poolers.py"]
            ]
        },
        "rows": [],
    }
    for name, lengths, sizes in cases:
        prompts = [texts[n] for n in lengths]
        params = [PoolingParams(late_chunk_size=c, use_activation=True) for c in sizes]

        def run(prompts=prompts, params=params):
            return llm.encode(
                prompts,
                pooling_task="token_embed",
                pooling_params=params,
                use_tqdm=False,
            )

        raw = llm.encode(
            prompts,
            pooling_task="token_embed",
            pooling_params=PoolingParams(use_activation=False),
            use_tqdm=False,
        )
        actual = run()
        minimum_cosine = 1.0
        for r, output, size in zip(raw, actual, sizes):
            states = r.outputs.data
            reference = (
                states
                if size is None
                else torch.stack([part.float().mean(0) for part in states.split(size)])
            )
            reference = F.normalize(reference, dim=-1)
            assert output.prompt_token_ids == r.prompt_token_ids
            assert output.outputs.data.shape == reference.shape
            similarity = (
                F.cosine_similarity(
                    output.outputs.data.float(), reference.float(), dim=-1
                )
                .min()
                .item()
            )
            assert similarity >= 0.999, (name, similarity)
            minimum_cosine = min(minimum_cosine, similarity)
        for _ in range(5):
            run()
        elapsed = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            run()
            elapsed.append((time.perf_counter() - start) * 1000)
        row = {
            "case": name,
            "lengths": lengths,
            "chunk_sizes": sizes,
            "minimum_cosine": minimum_cosine,
            "median_ms": statistics.median(elapsed),
            "p10_ms": float(torch.tensor(elapsed).quantile(0.1)),
            "p90_ms": float(torch.tensor(elapsed).quantile(0.9)),
            "samples_ms": elapsed,
        }
        result["rows"].append(row)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(
            json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--output", default="late-chunking-end-to-end.json")
    main(parser.parse_args())
