"""Independent full-forward reference for one response from every prompt group."""
import json
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

root = Path(__file__).resolve().parent
outdir = root / 'hf-reference'
outdir.mkdir(exist_ok=False)
torch.set_num_threads(4)
model = AutoModelForCausalLM.from_pretrained(
    root / 'model', dtype=torch.bfloat16, attn_implementation='eager',
    local_files_only=True,
).to('cuda:0').eval()
data = json.loads((root/'generated.json').read_text())
records = []
for group in data['groups']:
    p = len(group['prompt_token_ids'])
    response = group['responses'][0]['ids']
    ids = group['prompt_token_ids'] + response
    with torch.inference_mode():
        output = model(torch.tensor([ids], device='cuda:0'), use_cache=False)
        logits = output.logits[0]
        targets = torch.tensor(response, device='cuda:0').unsqueeze(-1)
        aligned = logits[p-1:len(ids)-1].float().log_softmax(-1).gather(-1, targets).cpu().numpy()
        shifted = logits[p:len(ids)].float().log_softmax(-1).gather(-1, targets).cpu().numpy()
    np.savez(outdir/f"{group['group']}.npz",aligned=aligned,shifted=shifted)
    comparisons = {}
    for folder in sorted(root.glob('v2-tp*')):
        if not folder.is_dir():
            continue
        path = folder/f"{group['group']}-0.npz"
        if not path.exists():
            continue
        with np.load(path) as arrays:
            comparisons[folder.name] = {}
            for mode in ['baseline','bounded']:
                a = arrays[mode]
                diff = np.abs(a-aligned)
                wrong = np.abs(a-shifted)
                comparisons[folder.name][mode] = {
                    'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
                    'p95_abs':float(np.quantile(diff,.95)),
                    'shift_one_row_mean_abs':float(wrong.mean()),
                }
    records.append({'group':group['group'],'p':p,'r':len(response),'comparisons':comparisons})
    (outdir/'results.json').write_text(json.dumps({'torch':torch.__version__,
        'transformers':transformers.__version__,'dtype':'bfloat16','attention':'eager',
        'scope':'Full-forward scores for first response of each group. Differences recorded, no universal training tolerance inferred.',
        'cases':records},indent=2))
    print(group['group'], 'done', flush=True)
    del output, logits
