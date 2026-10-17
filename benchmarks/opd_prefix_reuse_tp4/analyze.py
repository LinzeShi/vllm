"""Validate the downloaded records and regenerate the public timing table."""
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = json.loads((ROOT / 'repro/generated.json').read_text())
GROUPS = {g['group']: g for g in INPUT['groups']}
summary = {'rows': [], 'numerics': {}, 'validation': {}}
for budget in [512, 2048]:
    directory = ROOT / f'cloud/repro/tp4-32b-b{budget}-bf16'
    records = [json.loads(s) for s in (directory / 'results.jsonl').read_text().splitlines()]
    assert records[-1]['type'] == 'completion'
    manifest = records[0]
    assert manifest['args']['tp'] == 4 and manifest['args']['performance']
    assert manifest['model_revision'] == '9216db5781bf21249d130ec9da846c4624c16137'
    assert manifest['batch_invariant'] == '0' and not manifest['eager']
    assert manifest['input_sha256'] == hashlib.sha256((ROOT / 'repro/generated.json').read_bytes()).hexdigest()
    assert manifest['probe_sha256'] == hashlib.sha256((ROOT / 'repro/probe.py').read_bytes()).hexdigest()
    timings = [x for x in records if x['type'] == 'real_timing']
    numerics = [x for x in records if x['type'] == 'real_numerics']
    boundaries = [x for x in records if x['type'] == 'boundary']
    assert len(timings) == 16 and len(numerics) == 8 and len(boundaries) == 6
    assert len({(x['group'], x['arrival']) for x in timings}) == 16
    all_delta = {key: [] for key in ['reuse', 'single_vs_batch', 'repeat']}
    for row in numerics:
        group = GROUPS[row['group']]
        for i, response in enumerate(group['responses']):
            data = np.load(directory / f"{row['group']}-{i}.npz")
            assert int(data['p']) == len(group['prompt_token_ids'])
            assert data['ids'].tolist() == group['prompt_token_ids'] + response['ids']
            for key in ['baseline', 'bounded', 'single']:
                assert data[key].shape == (len(response['ids']), 1)
                assert np.isfinite(data[key]).all()
            all_delta['reuse'].extend(np.abs(data['baseline'] - data['bounded']).ravel().tolist())
            all_delta['single_vs_batch'].extend(np.abs(data['baseline'] - data['single']).ravel().tolist())
        all_delta['repeat'].extend(d['max_abs'] for d in row['repeat_difference'])
    timing_max = {'baseline': 0.0, 'bounded': 0.0}
    for row in timings:
        group = GROUPS[row['group']]
        for mode, measurements in row['measurements'].items():
            assert len(measurements) == 6
            median = statistics.median(x['seconds'] for x in measurements)
            assert abs(median - row['medians'][mode]) < 1e-10
            for rep, measurement in enumerate(measurements):
                assert measurement['seconds'] > 0 and measurement['hits'][0] == 0
                if mode == 'baseline':
                    assert measurement['hits'] == [0, 0, 0, 0]
                assert all(0 <= h <= len(group['prompt_token_ids']) - 1 for h in measurement['hits'])
                assert all(0 <= x['hit'] <= x['cap'] for x in measurement['lookups'])
                assert all(x['cap'] == len(group['prompt_token_ids']) - 1 for x in measurement['lookups'])
                assert all(x['skip'] == (mode == 'baseline') for x in measurement['lookups'])
                archive = np.load(directory / f"timing-{row['group']}-{row['arrival']}-{rep}-{mode}.npz")
                for i, response in enumerate(group['responses']):
                    a = archive[f'response_{i}']
                    assert a.shape == (len(response['ids']), 1) and np.isfinite(a).all()
                    reference = np.load(directory / f"{row['group']}-{i}.npz")['baseline']
                    delta = float(np.max(np.abs(a - reference)))
                    assert abs(delta - measurement['vs_reference'][i]['max_abs']) < 1e-7
                    timing_max[mode] = max(timing_max[mode], delta)
    summary['numerics'][str(budget)] = {
        key: {'max_abs': max(values), 'mean_abs': statistics.mean(values),
              'p50_abs': float(np.quantile(values, .5)),
              'p95_abs': float(np.quantile(values, .95)),
              'p99_abs': float(np.quantile(values, .99))}
        for key, values in all_delta.items()
    }
    summary['numerics'][str(budget)]['timing_max_vs_group_reference'] = timing_max
    summary['numerics'][str(budget)]['boundary_max_abs'] = max(x['difference']['max_abs'] for x in boundaries)
    summary['numerics'][str(budget)]['preemption'] = [x for x in records if x['type'] == 'forced_preemption']
    summary['validation'][str(budget)] = {'response_matrices': 32, 'timed_matrices': 768,
        'boundary_cases': 6, 'completed': True}
    for workload in ['math', 'qasper']:
        for arrival in ['group', 'staggered']:
            selected = [x for x in timings if x['group'].startswith(workload) and x['arrival'] == arrival]
            assert len(selected) == 4
            medians = {m: sum(x['medians'][m] for x in selected) for m in ['baseline', 'bounded']}
            # Per-repetition aggregates expose variation; they are not an independent-sample CI.
            repetitions = {m: [sum(x['measurements'][m][rep]['seconds'] for x in selected)
                               for rep in range(6)] for m in medians}
            response_tokens = sum(sum(len(r['ids']) for r in GROUPS[x['group']]['responses']) for x in selected)
            summary['rows'].append({'budget': budget, 'workload': workload, 'arrival': arrival,
                **medians, 'speedup': medians['baseline']/medians['bounded'],
                'time_saved_fraction': 1-medians['bounded']/medians['baseline'],
                'response_tokens': response_tokens,
                'scored_tokens_per_second': {m: response_tokens/t for m, t in medians.items()},
                'aggregate_repetitions': repetitions,
                'per_group_speedups': {x['group']: x['speedup'] for x in selected}})
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))
lines = ['| Token budget | Workload | Arrival | Uncached | Bounded reuse | Speedup |',
         '| --- | --- | --- | --- | --- | --- |']
for row in summary['rows']:
    workload = 'Short math prompts' if row['workload'] == 'math' else 'Long documents'
    arrival = 'Grouped' if row['arrival'] == 'group' else 'Staggered'
    lines.append(f"| {row['budget']} | {workload} | {arrival} | {row['baseline']:.3f} s | {row['bounded']:.3f} s | {row['speedup']:.3f}× |")
(ROOT / 'performance-table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(json.dumps(summary['numerics'], indent=2))
