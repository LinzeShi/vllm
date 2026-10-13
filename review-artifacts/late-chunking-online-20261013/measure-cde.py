"""Local validation aid, kept outside the upstream patch."""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

MODEL = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
CASES = [(8, 1, 16), (40, 1, 32), (72, 4, 64), (8, 1, 1)]


def inputs(repeats, batch):
    return ['passage: ' + 'Berlin has many museums. ' * repeats for _ in range(batch)]


def reduce_tokens(data, offsets, size):
    vectors, chunks = [], []
    states = np.asarray(data, dtype=np.float32)
    for start in range(0, len(states), size):
        end = min(start + size, len(states))
        vector = states[start:end].mean(axis=0, dtype=np.float32)
        vectors.append(vector / np.linalg.norm(vector))
        source = [(a, b) for a, b in offsets[start:end] if a < b]
        chunks.append({'token_range': [start, end], 'char_range':
                       [min(a for a, b in source), max(b for a, b in source)] if source else None})
    return np.asarray(vectors), chunks


def summary(samples):
    return {'runs': len(samples), 'median_ms': statistics.median(samples),
            'p95_ms': float(np.percentile(samples, 95)),
            'stdev_ms': statistics.stdev(samples), 'samples_ms': samples}


def online(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    records = []
    with requests.Session() as session:
        for repeats, batch, size in CASES:
            texts = inputs(repeats, batch)
            count = len(tokenizer(texts[0])['input_ids'])
            assert count <= 512
            timings = {'native': [], 'client': []}
            wire = {}
            max_abs = 0.0
            minimum_cosine = 1.0
            def call(mode):
                start = time.perf_counter()
                offsets = ([tokenizer(t, return_offsets_mapping=True)['offset_mapping']
                            for t in texts] if mode == 'client' else None)
                body = {'model': MODEL, 'task': 'token_embed', 'input': texts,
                        'use_activation': mode == 'native', 'encoding_format': 'float'}
                if mode == 'native':
                    body['late_chunking_params'] = {'chunk_size': size}
                response = session.post(args.url.rstrip('/') + '/pooling', json=body, timeout=120)
                response.raise_for_status()
                result = response.json()
                assert result['usage']['prompt_tokens'] == count * batch
                if mode == 'native':
                    reduced = [(np.asarray(item['data'], dtype=np.float32),
                                item['late_chunking']['chunks']) for item in result['data']]
                else:
                    reduced = [reduce_tokens(item['data'], off, size)
                               for item, off in zip(result['data'], offsets)]
                return (time.perf_counter() - start) * 1000, len(response.content), reduced
            for run in range(args.runs + 2):
                outputs = {}
                for mode in (('client', 'native') if run % 2 else ('native', 'client')):
                    elapsed, wire[mode], outputs[mode] = call(mode)
                    if run >= 2:
                        timings[mode].append(elapsed)
                for (got, ranges), (reference, expected_ranges) in zip(outputs['native'], outputs['client']):
                    assert ranges == expected_ranges
                    cosine = np.sum(got * reference, axis=-1) / (np.linalg.norm(got,axis=-1) * np.linalg.norm(reference,axis=-1))
                    minimum_cosine = min(minimum_cosine, float(cosine.min()))
                    max_abs = max(max_abs, float(np.abs(got-reference).max()))
                    assert np.all(cosine >= .999)
                    np.testing.assert_allclose(np.linalg.norm(got,axis=-1), 1, atol=.001)
            records.append({'input_tokens_per_doc': count, 'batch': batch, 'chunk_size': size,
                            'json_response_bytes': wire, 'native': summary(timings['native']),
                            'client': summary(timings['client']), 'minimum_cosine': minimum_cosine,
                            'max_absolute_error': max_abs})
    return {'kind': 'same-pod loopback HTTP end-to-end, includes client offsets/reduction and JSON',
            'model': MODEL, 'revision': REVISION, 'dtype': 'bfloat16',
            'warmups_per_mode': 2, 'records': records}


def reset_peak(worker):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_memory(worker):
    import torch
    torch.cuda.synchronize()
    return {'allocated': torch.cuda.max_memory_allocated(), 'reserved': torch.cuda.max_memory_reserved()}


def memory(args):
    from vllm import LLM, PoolingParams
    from vllm.config import PoolerConfig
    from vllm.pooling_params import LateChunkingParams
    llm = LLM(model=MODEL, revision=REVISION, runner='pooling', dtype='bfloat16',
              pooler_config=PoolerConfig(task='token_embed'), max_model_len=512,
              gpu_memory_utilization=.5, enforce_eager=True)
    records = []
    for repeats, batch, size in CASES:
        texts = inputs(repeats, batch)
        for mode in ('client', 'native'):
            params = PoolingParams(use_activation=mode == 'native',
                                   late_chunking_params=LateChunkingParams(size) if mode == 'native' else None)
            llm.encode(texts, pooling_task='token_embed', pooling_params=params, use_tqdm=False)
            peaks = []
            for _ in range(3):
                llm.collective_rpc(reset_peak)
                outputs = llm.encode(texts, pooling_task='token_embed', pooling_params=params, use_tqdm=False)
                peaks.append(llm.collective_rpc(peak_memory)[0])
            records.append({'input_tokens_per_doc': len(outputs[0].prompt_token_ids), 'batch':batch,
                            'chunk_size':size,'mode':mode,'peaks_bytes':peaks})
    return {'kind':'offline worker CUDA allocator peak, separate from HTTP timings; both modes warmed',
            'model':MODEL,'revision':REVISION,'dtype':'bfloat16','records':records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['online','memory'])
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--runs', type=int, default=9)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = online(args) if args.mode == 'online' else memory(args)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
