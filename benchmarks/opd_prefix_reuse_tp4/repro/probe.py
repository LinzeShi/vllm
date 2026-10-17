"""Production scorer validation. Lookup instrumentation only records results."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager

ROOT = Path(__file__).resolve().parent
MODE = 'baseline'
ORIGINAL = KVCacheManager.get_computed_blocks
LOOKUPS = []


def observe_lookup(self, request):
    out = ORIGINAL(self, request)
    params = request.sampling_params
    if params is not None and params.prompt_logprob_token_ids is not None:
        LOOKUPS.append({'id': request.request_id,
                        'cap': request.max_cache_hit_length,
                        'hit': out[1], 'preemptions': request.num_preemptions,
                        'skip': request.skip_reading_prefix_cache})
    return out


KVCacheManager.get_computed_blocks = observe_lookup


def differences(a, b):
    assert a.shape == b.shape
    assert not np.isnan(a).any() and not np.isnan(b).any()
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    assert np.array_equal(np.isneginf(a), np.isneginf(b))
    keep = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[keep] - b[keep])
    return {'max_abs': float(d.max()) if d.size else 0,
            'mean_abs': float(d.mean()) if d.size else 0,
            'exact': bool(np.array_equal(a, b))}


def main():
    global MODE
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['generate', 'evaluate'], required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--label', required=True)
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--disable-custom-all-reduce', action='store_true')
    parser.add_argument('--response-cap', type=int, default=1024)
    args = parser.parse_args()
    outdir = ROOT / args.label
    outdir.mkdir(exist_ok=False)
    def record(obj):
        with (outdir / 'results.jsonl').open('a') as f:
            f.write(json.dumps(obj) + '\n')
        print(json.dumps(obj), flush=True)
    record({'type': 'manifest', 'args': vars(args), 'vllm': vllm.__version__,
            'torch': torch.__version__, 'batch_invariant': os.getenv('VLLM_BATCH_INVARIANT', '0'),
            'model': 'Qwen/Qwen3-32B', 'model_revision': '9216db5781bf21249d130ec9da846c4624c16137',
            'source_base': '90d6d871d6700dd229a76b788bf0dca22bab6c70',
            'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_sha256': hashlib.sha256((ROOT / ('generated.json' if args.phase == 'evaluate' else 'prompts.json')).read_bytes()).hexdigest(),
            'environment': {k: os.getenv(k) for k in ('NCCL_P2P_DISABLE', 'VLLM_GPU_SYNC_CHECK', 'VLLM_WORKER_MULTIPROC_METHOD')},
            'async_scheduling': False, 'eager': False, 'num_gpu_blocks': 2048})
    llm = LLM(model='Qwen/Qwen3-32B', revision='9216db5781bf21249d130ec9da846c4624c16137', dtype='bfloat16', max_model_len=8192,
              tensor_parallel_size=args.tp, disable_custom_all_reduce=args.disable_custom_all_reduce, distributed_executor_backend='mp' if args.tp > 1 else 'uni',
              max_num_seqs=4, max_num_batched_tokens=args.budget, block_size=16,
              num_gpu_blocks_override=2048, gpu_memory_utilization=.80,
              enforce_eager=False, enable_prefix_caching=True, async_scheduling=False,
              logprobs_mode='raw_logprobs', seed=31)
    counter = 0

    def params(ids, p, mode, ragged=False, ordinary=False):
        rows = [[t, (t + 17) % 151000] if j % 2 == 0 else [t]
                for j, t in enumerate(ids[p:])] if ragged else [[t] for t in ids[p:]]
        kw = dict(temperature=0, max_tokens=1, prompt_logprob_start=p-1,
                  prompt_logprob_token_ids=rows, detokenize=False)
        if mode != 'baseline':
            kw['skip_reading_prefix_cache'] = False
        if ordinary:
            kw['prompt_logprobs'] = 0
        return SamplingParams(**kw)

    def run(items, mode, arrival='group', ragged=False, ordinary=False):
        nonlocal counter
        global MODE
        MODE = mode
        prompts = [{'prompt_token_ids': x['ids']} for x in items]
        sampling = [params(x['ids'], x['p'], mode, ragged, ordinary) for x in items]
        before = len(LOOKUPS)
        begin = time.perf_counter()
        if arrival == 'group':
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
        else:
            engine = llm.llm_engine
            pending = list(zip(prompts, sampling))
            expected, done, arrivals = [], {}, []
            counter += 1
            # Fixed 20 ms arrival gaps; step execution can delay admission.
            while pending or engine.has_unfinished_requests():
                elapsed = time.perf_counter() - begin
                while pending and elapsed >= len(expected) * .02:
                    p, s = pending.pop(0)
                    rid = f'stagger-{counter}-{len(expected)}'
                    engine.add_request(rid, p, s)
                    expected.append(rid)
                    arrivals.append(elapsed)
                if engine.has_unfinished_requests():
                    for out in engine.step():
                        if out.finished:
                            done[out.request_id] = out
                elif pending:
                    time.sleep(.001)
            outputs = [done[rid] for rid in expected]
        elapsed = time.perf_counter() - begin
        arrays = []
        for item, out in zip(items, outputs, strict=True):
            if mode == 'unsafe' and out.prompt_token_id_logprobs is None:
                arrays.append(None)
                continue
            a = np.asarray(out.prompt_token_id_logprobs)
            assert a.shape == (len(item['ids'])-item['p'], 2 if ragged else 1), a.shape
            assert np.isfinite(a[:, 0]).all()
            if mode == 'bounded':
                assert out.num_cached_tokens <= item['p'] - 1
            arrays.append(a)
        assert all(x['hit'] <= x['cap'] for x in LOOKUPS[before:])
        if mode == 'baseline':
            assert all(o.num_cached_tokens == 0 for o in outputs)
        return arrays, {'seconds': elapsed, 'hits': [o.num_cached_tokens for o in outputs],
                        'lookups': LOOKUPS[before:], 'arrivals': arrivals if arrival != 'group' else [0]*len(items)}

    try:
        if args.phase == 'generate':
            data = json.loads((ROOT / 'prompts.json').read_text())
            for group in data['groups']:
                prompt = group['prompt_token_ids']
                sampling = [SamplingParams(temperature=.8, top_p=.95, max_tokens=args.response_cap,
                                            seed=3100+i, detokenize=True) for i in range(4)]
                out = llm.generate([{'prompt_token_ids': prompt}]*4, sampling, use_tqdm=False)
                group['responses'] = [{'ids': list(o.outputs[0].token_ids),
                                       'text': o.outputs[0].text,
                                       'finish_reason': o.outputs[0].finish_reason} for o in out]
                record({'type': 'generated', 'group': group['group'], 'p': len(prompt),
                        'r': [len(x['ids']) for x in group['responses']],
                        'finish': [x['finish_reason'] for x in group['responses']]})
                (ROOT / 'generated.json').write_text(json.dumps(data, indent=2))
            record({'type': 'completion'})
            return

        tok = llm.get_tokenizer()
        q = tok.encode('Explain the parity of the sum of two even integers. ', add_special_tokens=False)
        a = tok.encode('Each integer is twice another integer, and their sum is even. ', add_special_tokens=False)
        for p in [1, 16, 17, 33, 512, 2048]:
            r = 64
            ids = (q*(p//len(q)+1))[:p] + (a*(r//len(a)+1))[:r]
            item = {'ids': ids, 'p': p}
            assert llm.reset_prefix_cache() is not False
            base, _ = run([item], 'baseline', ragged=True)
            bounded, info = run([item], 'bounded', ragged=True)
            record({'type': 'boundary', 'p': p, 'r': r, 'bounded': info,
                    'difference': differences(bounded[0], base[0])})
            np.savez(outdir / f'boundary-{p}.npz', ids=np.array(ids), baseline=base[0], bounded=bounded[0])
        # Both output families must force a zero-length cache hit.
        both, both_info = run([item], 'bounded', ordinary=True)
        assert both_info['hits'] == [0]
        record({'type': 'both_outputs', **both_info})

        # Inject a synchronous scheduler preemption after scored rows exist.
        p, r = 64, 1536
        ids = (q*(p//len(q)+1))[:p] + (a*(r//len(a)+1))[:r]
        item = {'ids': ids, 'p': p}
        assert llm.reset_prefix_cache() is not False
        base, _ = run([item], 'baseline')
        MODE = 'bounded'
        before = len(LOOKUPS)
        engine = llm.llm_engine
        engine.add_request('forced-prefill-preemption', {'prompt_token_ids': ids}, params(ids,p,'bounded'))
        engine.step()
        scheduler = engine.engine_core.engine_core.scheduler
        req = next((x for x in scheduler.running if x.num_computed_tokens > p-1), None)
        progress = req.num_computed_tokens if req is not None else len(ids)
        if req is not None and progress < len(ids):
            scheduler.running.remove(req)
            scheduler._preempt_request(req, time.monotonic())
            finished = []
            while engine.has_unfinished_requests():
                finished.extend(x for x in engine.step() if x.finished)
            out = finished[-1]
            scores = np.asarray(out.prompt_token_id_logprobs)
            assert scores.shape == (r,1)
            record({'type': 'forced_preemption', 'progress': progress, 'start': p-1,
                    'lookups': LOOKUPS[before:], 'difference': differences(scores,base[0])})
            np.savez(outdir/'preemption.npz',baseline=base[0],bounded=scores)
        else:
            record({'type': 'forced_preemption', 'status': 'not_applicable_budget_finishes_prefill'})
            while engine.has_unfinished_requests():
                engine.step()

        data = json.loads((ROOT / 'generated.json').read_text())
        for group in data['groups']:
            p = len(group['prompt_token_ids'])
            items = [{'ids': group['prompt_token_ids'] + x['ids'], 'p': p} for x in group['responses']]
            assert llm.reset_prefix_cache() is not False
            reference, _ = run(items, 'baseline')
            same_batch_repeat, _ = run(items, 'baseline')
            singles = [run([item], 'baseline')[0][0] for item in items]
            bounded, info = run(items, 'bounded')
            record({'type': 'real_numerics', 'group':group['group'], 'p':p,
                    'r':[len(i['ids'])-p for i in items], 'bounded_info':info,
                    'reuse_difference':[differences(x,y) for x,y in zip(bounded,reference)],
                    'batch_difference':[differences(x,y) for x,y in zip(singles,reference)],
                    'repeat_difference':[differences(x,y) for x,y in zip(same_batch_repeat,reference)]})
            for i in range(4):
                np.savez(outdir/f"{group['group']}-{i}.npz",ids=np.array(items[i]['ids']),p=p,
                         baseline=reference[i],bounded=bounded[i],single=singles[i])
            if not args.performance:
                continue
            for arrival in ['group','staggered']:
                # Warm both paths and arrival patterns before any timed repetition.
                for warm_mode in ['baseline', 'bounded']:
                    assert llm.reset_prefix_cache() is not False
                    run(items, warm_mode, arrival)
                measurements={'baseline':[], 'bounded':[]}
                for rep in range(6):
                    for mode in (['baseline','bounded'] if rep%2==0 else ['bounded','baseline']):
                        assert llm.reset_prefix_cache() is not False
                        scores, timing = run(items,mode,arrival)
                        assert timing['hits'][0] == 0, 'Each prompt group must start cold'
                        timing['vs_reference'] = [differences(x, y) for x, y in zip(scores, reference)]
                        timing['response_tokens'] = sum(len(x['ids'])-p for x in items)
                        np.savez_compressed(outdir/f"timing-{group['group']}-{arrival}-{rep}-{mode}.npz", **{f'response_{i}': a for i, a in enumerate(scores)})
                        measurements[mode].append(timing)
                medians={k:statistics.median(x['seconds'] for x in v) for k,v in measurements.items()}
                record({'type':'real_timing','group':group['group'],'arrival':arrival,
                        'measurements':measurements,'medians':medians,
                        'speedup':medians['baseline']/medians['bounded']})
        record({'type': 'completion'})
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
