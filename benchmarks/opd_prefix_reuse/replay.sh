#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
for spec in '1 512 0 yes' '1 512 1 yes' '1 2048 0 yes' '1 2048 1 yes' '2 512 0 yes' '2 512 1 no' '4 512 0 yes' '4 512 1 no'; do
    read -r tp budget invariant perf <<< "$spec"
    label="v2-tp${tp}-b${budget}-bi${invariant}"
    export VLLM_BATCH_INVARIANT="$invariant"
    extra=()
    if test "$perf" = yes; then extra=(--performance); fi
    .venv/bin/python probe.py --phase evaluate --tp "$tp" --budget "$budget" --label "$label" "${extra[@]}" > "$label.log" 2>&1
    echo 0 > "$label.exit"
done
.venv/bin/python hf_reference.py
