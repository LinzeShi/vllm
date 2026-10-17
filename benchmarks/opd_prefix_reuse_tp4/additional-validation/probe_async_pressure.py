"""Local audit: async scheduling, bounded hits and forced prefill preemption."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

import numpy as np

from vllm import LLM, SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

HERE = Path(__file__).resolve().parent
OUT = HERE / sys.argv[1]
OUT.mkdir(exist_ok=False)
MODE = "setup"
LOOKUPS = []
PREEMPTIONS = []
original_lookup = KVCacheManager.get_computed_blocks
original_preempt = Scheduler._preempt_request


def observe_lookup(self, request):
    result = original_lookup(self, request)
    if request.sampling_params.prompt_logprob_token_ids is not None:
        LOOKUPS.append({"mode": MODE, "id": request.request_id,
                        "hit": result[1], "cap": request.max_cache_hit_length,
                        "preemptions": request.num_preemptions,
                        "skip": request.skip_reading_prefix_cache})
    return result


def observe_preempt(self, request, *args, **kwargs):
    PREEMPTIONS.append({"mode": MODE, "id": request.request_id,
                       "progress": request.num_computed_tokens,
                       "start": request.sampling_params.prompt_logprob_start,
                       "in_flight_tokens": request.num_in_flight_tokens,
                       "output_tokens": request.num_output_tokens})
    return original_preempt(self, request, *args, **kwargs)


def main():
    global MODE
    KVCacheManager.get_computed_blocks = observe_lookup
    Scheduler._preempt_request = observe_preempt
    llm = LLM(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
        dtype="bfloat16", seed=31, max_model_len=512, max_num_seqs=4,
        max_num_batched_tokens=48, block_size=16, num_gpu_blocks_override=33,
        gpu_memory_utilization=0.25, enforce_eager=True,
        enable_prefix_caching=True, enable_chunked_prefill=True,
        async_scheduling=True,
    )
    try:
        assert llm.llm_engine.vllm_config.scheduler_config.async_scheduling is True
        core = llm.llm_engine.engine_core.engine_core
        assert "AsyncScheduler" in type(core.scheduler).__name__
        tok = llm.get_tokenizer()
        base = tok.encode("Read the shared context and score the following response. ", add_special_tokens=False)
        context = (base * 20)[:129]
        prompts, params = [], []
        for i in range(8):
            suffix = tok.encode(f"Response {i}: " + "The answer depends on the details in the context. " * 30, add_special_tokens=False)[:200]
            ids = context + suffix
            candidates = np.array([[t, (t+17) % 151000, -1] for t in suffix], dtype=np.int32)
            prompts.append({"prompt_token_ids": ids})
            params.append(SamplingParams(temperature=0, max_tokens=1,
                          prompt_logprob_start=len(context)-1,
                          prompt_logprob_token_ids=candidates, detokenize=False))
        arms = {}
        for mode in ("baseline", "bounded", "preempted"):
            MODE = mode
            assert llm.reset_prefix_cache() is not False
            # Warm only the unscored context; the full answers are not cached.
            llm.generate([{"prompt_token_ids": context}], SamplingParams(max_tokens=1), use_tqdm=False)
            if mode != "baseline":
                for p in params:
                    p.skip_reading_prefix_cache = False
            if mode != "preempted":
                outputs = llm.generate(prompts, params, use_tqdm=False)
            else:
                engine = llm.llm_engine
                ids = [f"async-preempt-{i}" for i in range(len(prompts))]
                for rid, prompt, sampling in zip(ids, prompts, params, strict=True):
                    engine.add_request(rid, prompt, sampling)
                done = {}
                injected = False
                while engine.has_unfinished_requests():
                    for output in engine.step():
                        if output.finished:
                            done[output.request_id] = output
                    if not injected:
                        request = next((r for r in core.scheduler.running
                            if r.sampling_params.prompt_logprob_start + 96
                            <= r.num_computed_tokens < r.num_prompt_tokens
                            and r.num_output_tokens == 0), None)
                        if request is not None:
                            core.scheduler.running.remove(request)
                            core.scheduler._preempt_request(request, time.monotonic())
                            injected = True
                assert injected, "Must inject a preemption inside scored prefill"
                outputs = [done[rid] for rid in ids]
            arrays = [np.asarray(x.prompt_token_id_logprobs) for x in outputs]
            for p, a in zip(params, arrays, strict=True):
                assert a.shape == p.prompt_logprob_token_ids.shape == (200, 3)
                assert np.isfinite(a[:, :2]).all()
                assert np.isneginf(a[:, 2]).all()
            arms[mode] = arrays
        assert all(x["hit"] <= x["cap"] == 128 for x in LOOKUPS)
        assert all(x["skip"] == (x["mode"] == "baseline") for x in LOOKUPS)
        resumed = [x for x in LOOKUPS if x["mode"] == "preempted" and x["preemptions"] > 0]
        meaningful = [x for x in PREEMPTIONS if x["mode"] == "preempted" and x["progress"] > x["start"] and x["output_tokens"] == 0]
        (OUT / "async-pressure-events.json").write_text(json.dumps({"lookups":LOOKUPS,"preemptions":PREEMPTIONS},indent=2)+"\n")
        assert resumed and meaningful, "Must actually preempt and resume scored prefill"
        assert any(x["hit"] > 0 for x in resumed), "Resume must exercise real bounded cache hits"
        differences = {}
        for left, right in [("baseline", "bounded"), ("baseline", "preempted"), ("bounded", "preempted")]:
            delta = np.concatenate([np.abs(a[:, :2] - b[:, :2]) for a,b in zip(arms[left],arms[right],strict=True)])
            differences[f"{left}_vs_{right}"] = {
                "max_actual_token":float(delta[:,0].max()),
                "max_extra_candidate":float(delta[:,1].max()),
                "mean_actual_token":float(delta[:,0].mean()),
                "mean_extra_candidate":float(delta[:,1].mean()),
                "exact":bool((delta == 0).all()),
            }
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
            assert all(x["exact"] for x in differences.values()), differences
        result = {"scheduler":type(core.scheduler).__name__,"async_scheduling":True,
                  "strategy":"forced scheduler preemption inside scored prefill",
                  "model_revision":"c1899de289a04d12100db370d81485cdf75e47ca",
                  "execution":"eager", "sync_check":os.environ.get("VLLM_GPU_SYNC_CHECK"),
                  "outputs_per_path":8,"rows_per_output":200,
                  "batch_invariant":os.environ.get("VLLM_BATCH_INVARIANT", "0"),
                  "differences":differences,
                  "lookups":LOOKUPS,"preemptions":PREEMPTIONS,
                  "bounded_prefill_preemptions_after_start":len(meaningful),
                  "bounded_resumed_lookups":len(resumed),"ok":True}
        np.savez_compressed(OUT / "async-pressure-arrays.npz", **{f"{mode}_{i}":a for mode, arrays in arms.items() for i,a in enumerate(arrays)})
        (OUT / "async-pressure-result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ['lookups','preemptions']}),flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
