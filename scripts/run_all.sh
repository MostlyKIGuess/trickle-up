#!/usr/bin/env bash
# Regenerate every deliverable from the model file, in order, with the fixed seeds of each script.
#
# Usage (from anywhere): scripts/run_all.sh [spice_workers]
#
# Needs Nix with flakes. SPICE, composition and plots run in the default shell, the training in the
# GPU shell (CUDA PyTorch). Writes data/ (gitignored), results/, models/, spice/ and report/analysis/.
# Each step's output goes to data/logs/<step>.log as well as the terminal. A failing step stops the run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${1:-8}"
cd "$ROOT"
mkdir -p data/logs
git rev-parse HEAD > data/logs/git_sha.txt
git status --porcelain --untracked-files=no > data/logs/git_status.txt

step() {
    local name="$1" shell="$2"
    shift 2
    echo "== $name: $*"
    nix develop "$ROOT#$shell" -c "$@" 2>&1 | tee "data/logs/$name.log"
}

# Part A: transistor data, data check and trends.
step 01_sample default python3 scripts/01_sample_fet.py
step 02_sim default python3 scripts/02_sim_fet.py all "$WORKERS"
# The four datasets of the brief are a deliverable, so they are kept in git (data/ is not).
mkdir -p datasets
cp data/fet_nmos_on.csv data/fet_nmos_off.csv data/fet_pmos_on.csv data/fet_pmos_off.csv datasets/
step 03_check default python3 scripts/03_check_data.py
step 10_trends default python3 scripts/10_trend_sweeps.py

# Part A: the comparison of the four model types and two transforms with the learning curve
# (results/comparison/), then the frozen method of config/stage4_frozen.toml (results/, models/). The
# uniform-only run is for the error analysis of the report only, so it goes to data/.
step 04_compare gpu python3 scripts/04_train.py --output data/comparison
rm -rf results/comparison
cp -r data/comparison/results results/comparison
step 04_train gpu python3 scripts/04_train.py --models mlp --transforms log --no-curve --seed 463 \
    --separate --mask-off-ig --ensemble 5
step 04_train_uniform_only gpu python3 scripts/04_train.py --models mlp --transforms log --no-curve \
    --training-sets uniform --output data/uniform_only

# Part B: corners, gate SPICE, then the oracle control and the ML gate leakage.
step 05_corners default python3 scripts/05_corners.py
step 06_gates_spice default python3 scripts/06_gates_spice.py 200 "$WORKERS"
step 06_diffusion default python3 scripts/06_gates_spice.py diffusion 200 "$WORKERS"
step 07_gates_oracle default python3 scripts/07_gates_ml.py oracle
step 07_gates_ml default python3 scripts/07_gates_ml.py models

# Part C: adder SPICE, then the oracle control and the ML adder leakage, both from the Part B gates.
# 09 reads data/gate_predictions_<name>.csv, so each 09 runs after its 07.
step 08_cla_spice default python3 scripts/08_cla_spice.py 200 "$WORKERS"
step 09_cla_oracle default python3 scripts/09_cla_ml.py oracle
step 09_cla_ml default python3 scripts/09_cla_ml.py ml
# Time of one adder leakage with SPICE and with our method, single process (results/timing.csv).
step 13_timing default python3 scripts/13_timing.py

# Report tables and the error analysis.
step summarize_models default python3 report/summarize_models.py
step error_analysis default python3 report/error_analysis.py data models data/uniform_only/models
echo "== done, logs in data/logs/"
