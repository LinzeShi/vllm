"""Verify the published score arrays and print the timing ratios without a GPU."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
summary = json.loads((root / 'summary.json').read_text())
count = 0
for folder in sorted((root / 'results').glob('v2-tp*-bi1')):
    for path in folder.glob('*.npz'):
        with np.load(path) as arrays:
            assert np.array_equal(arrays['baseline'], arrays['bounded']), path
            count += 1
assert count == 155, count
print(f'{count} saved controlled comparisons match exactly.')
for budget in (512, 2048):
    run = summary['runs'][f'v2-tp1-b{budget}-bi1']
    for name, result in run['performance'].items():
        print(budget, name, f"{result['ratio_of_summed_medians']:.3f}x")
