#!/usr/bin/env bash
# Train the runs of config/stage3_arms.toml one after another, each into data/arms/<label>_s<seed>/.
#
# Usage: scripts/run_arms.sh <label> <seed> [<seed> ...] -- [04_train.py flags]
#   scripts/run_arms.sh baseline 463 464 465 466 467 468 469 470 --
#   scripts/run_arms.sh A1 463 464 465 -- --weff
#
# Every run is the MLP with the log target, no learning curve, selected by the median score as the
# committed models were. A run directory that already has models is skipped, so a stopped batch resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
label="$1"
shift
seeds=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    seeds+=("$1")
    shift
done
if [[ $# -eq 0 ]]; then
    echo "missing -- before the 04_train.py flags" >&2
    exit 2
fi
shift
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "uncommitted changes to tracked files, commit first so the runs are pinned" >&2
    exit 2
fi
for seed in "${seeds[@]}"; do
    out="data/arms/${label}_s${seed}"
    if [[ -f "$out/models/fet_pmos_off.joblib" ]]; then
        echo "== skip $out, already trained"
        continue
    fi
    mkdir -p "$out"
    git rev-parse HEAD > "$out/git_sha.txt"
    echo "== $out: $*"
    nix develop "$ROOT#gpu" -c python3 scripts/04_train.py --models mlp --transforms log --no-curve \
        --seed "$seed" --output "$out" "$@" > "$out/train.log" 2>&1
    nix develop "$ROOT#default" -c python3 scripts/11_compare_runs.py metrics "$out" > "$out/metrics.log" 2>&1
done
