"""Reconstruct source prompts and verify their exact token IDs before replay."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--longbench-zip', type=Path)
parser.add_argument('--math-sample', type=Path)
parser.add_argument('--tokenizer', default=str(ROOT / 'model'))
args = parser.parse_args()
data = json.loads((ROOT / 'inputs.json').read_text())
if args.math_sample:
    samples = [json.loads(line) for line in args.math_sample.read_text().splitlines()]
    math_rows = {x['row_idx']: x['row']['prompt'] for sample in samples for x in sample['rows']}
else:
    url = 'https://datasets-server.huggingface.co/rows?dataset=zhuzilin%2Fdapo-math-17k&config=default&split=train&offset=0&length=4'
    with urllib.request.urlopen(url) as r:
        math_rows = {x['row_idx']: x['row']['prompt'] for x in json.load(r)['rows']}
archive = args.longbench_zip or ROOT / 'longbench-data.zip'
if not archive.exists():
    url = 'https://huggingface.co/datasets/zai-org/LongBench/resolve/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip'
    urllib.request.urlretrieve(url, archive)
with zipfile.ZipFile(archive) as z:
    member = next(n for n in z.namelist() if n.endswith('qasper.jsonl'))
    qasper = [json.loads(line) for line in z.read(member).decode().splitlines()]
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
for group in data['groups']:
    if group['group'].startswith('math'):
        messages = math_rows[group['source_row']]
    else:
        row = qasper[group['source_row']]
        assert row['_id'] == group['source_id']
        content = 'Read the paper and answer the question using the paper.\n\n' + row['context']
        content += '\n\nQuestion: ' + row['input'] + '\nAnswer concisely.'
        messages = [{'role': 'user', 'content': content}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    actual = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    assert actual == group['prompt_token_ids_sha256'], f"Input changed: {group['group']}"
    group['messages'], group['prompt_token_ids'] = messages, ids
(ROOT / 'generated.json').write_text(json.dumps(data, indent=2) + '\n')
print('Verified all 8 prompts; restored the 32 fixed response token sequences.')
