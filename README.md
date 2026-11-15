# ML-driven PVT variation aware leakage estimation

ECE463 Digital VLSI Design course project. The brief and the TA material are in `given/`, the report
is `report/report.pdf`.

On 400 corners drawn after the method was frozen, the ML gate leakage is within 0.60 % of SPICE
(mean, 95 % interval 0.56 to 0.64 %) and the adder within 0.24 % (0.21 to 0.26 %), against 2.14 and
1.32 % for the first version's models on the same corners (`results/stage4/stage4_summary.csv`).

| Item | Value |
| --- | --- |
| Team | Dvd Player |
| Members | Aditya Peketi, Krish Pandya, Sudhanva Joshi, Vignesh Vembar |
| Model file | PTM 22nm LP (`given/models/22nm_LP.pm`) |
| Nominal VDD | 1 V |
| Test circuit | 3-bit carry-lookahead adder |

## Environment

```sh
nix develop          # ngspice, typst, numpy, pandas, scikit-learn, xgboost, lightgbm
nix develop .#gpu    # the same plus CUDA PyTorch, for training on the GPU machine
```

## Reproduce

Everything in the report comes from three sets of commands, run from the repository root.

```sh
# 1. Parts A, B and C, the report tables and the report figures (8 SPICE workers by default)
scripts/run_all.sh 8
# 2. the report
nix develop -c typst compile --root . report/report.typ report/report.pdf
```

`scripts/run_all.sh [spice_workers]` runs every step of the sections below in order with the fixed
seeds and logs each step to `data/logs/`. It writes `datasets/`, `results/`, `models/`, `spice/`
and `report/analysis/`. Its last two steps are `report/summarize_models.py` and
`report/error_analysis.py data models data/uniform_only/models`, which write the tables and figures of
`report/analysis/`. It does not rerun the `rgatemod` check, the width sweep, the arms or the held-out
evaluation, which have their own commands below. Appendix C of the report maps every deliverable of
the brief to its file.

`run_all.sh` trains the frozen method only. The runs that picked it are separate, since they are 29
trainings (`config/stage3_arms.toml`). Each goes to `data/arms/<label>_s<seed>/` and is scored on the
Part A validation rows as soon as it finishes.

```sh
scripts/run_arms.sh baseline 463 464 465 466 467 468 469 470 --
scripts/run_arms.sh A1 463 464 465 -- --weff
scripts/run_arms.sh A2 463 464 465 -- --separate
scripts/run_arms.sh A3 463 464 465 -- --training-sets uniform,targeted,gatebias
scripts/run_arms.sh A4 463 464 465 -- --mask-off-ig
scripts/run_arms.sh A5 463 473 483 -- --ensemble 5
scripts/run_arms.sh A2A4 463 464 465 -- --separate --mask-off-ig
scripts/run_arms.sh A2A4A5 463 473 483 -- --separate --mask-off-ig --ensemble 5
nix develop -c python3 scripts/11_compare_runs.py report data/arms   # noise_floor.csv, arm_report.csv
```

`results/arms/` holds the two tables of the last command and, in `runs/`, the `results/arm_metrics.csv`
of every run as `<label>_s<seed>.csv`. `run_arms.sh` refuses to start with uncommitted changes to
tracked files.

The held-out evaluation (`results/stage4/`) ran in a second checkout, so that the Part B data stays
as it is. There `CORNER_STREAM` in `scripts/05_corners.py` is 5 instead of 4.

```sh
python3 scripts/05_corners.py 400           # 400 new corners
python3 scripts/06_gates_spice.py 400 8     # 7,200 gate simulations
python3 scripts/08_cla_spice.py 400 8       # 51,200 adder simulations
# for each method: 07 then 09, as in Parts B and C below
python3 scripts/12_stage4_eval.py data/stage4_out <data dir of the Part B corners>
```

The three methods are the frozen models (`models/`), the first version's models (`models/` of commit
1437b67) and the oracle. For each one, `data/gate_predictions_<name>.csv` and
`data/cla_predictions_<name>.csv` are copied to `data/stage4_out/<frozen|old|oracle>/` as
`gate_predictions.csv` and `cla_predictions.csv` before `12_stage4_eval.py` runs. It writes
`stage4_summary.csv` and `stage4_verdicts.csv` there.

An MLP fit depends on the GPU. The same seed on the same GPU gives the same models, and a different
GPU gives slightly different ones.

`data/` is gitignored. Two things from it are deliverables and are kept in git: the four datasets of
the brief (`datasets/fet_<case>.csv`, 10,000 rows each) and the record of every gate simulation with
its SPICE leakage, ML leakage, output voltage and internal node voltages
(`report/analysis/gate_records.csv`, written by `report/error_analysis.py`).

## Part A: transistor data

```sh
python3 scripts/01_sample_fet.py 10000 5000 10000   # uniform, targeted and gate-bias sets
python3 scripts/02_sim_fet.py all                   # all three sets
```

Currents are positive into the transistor terminal, so `id + ig + is + ib = 0`. `id` and `is` are
the terminal source currents. `ig` and `ib` are the device's sums of its gate and body components,
because the gate and body source currents of a node held at VDD have a floating-point floor near
1e-16 A (the model has internal gate and body resistor nodes). Where the currents are large, the
device sums and the source currents agree to 5 digits. The decks print 12 significant digits
(`numdgt = 12`). The Kirchhoff residual against the largest terminal current of each sample is in
`results/kirchhoff.csv` (`python3 scripts/03_check_data.py`).

Three sample sets per case. `data/fet_<case>.csv` is the brief's dataset, every input uniform (copied
to `datasets/`). Inside a
gate, though, an ON transistor has drain and source at the same rail, |Vds| < 1 mV in 90 % of the
Part B simulations (median about 1 uV), where independent uniform Vd and Vs almost never land. Models
trained on the uniform set alone reached 0.4 to 1.3 % median error on its test rows, yet missed the
gate and source currents of those ON transistors by 40 to 400 % (median), and the gate leakage by
66 % (the first training runs, whose logs are not in this repository). `data/fet_<case>_targeted.csv` (5,000 rows) draws VDD,
temperature, width and process the same way, with Vd and Vs near each other or at the rails (the
docstring of `01_sample_fet.py`). Training uses both, and the test error is reported for each set.

`data/fet_<case>_gatebias.csv` (10,000 rows) draws the terminal voltages a transistor sees inside a
static gate, from the circuit and not from the Part B simulations. It is only used to score models
(the gate-bias score of `04_train.py`), never for training the frozen models.

Every transistor in every netlist gets source and drain diffusions `AD = AS = W * Ldiff` and
`PD = PS = 2 * (W + Ldiff)`, with `Ldiff = 44n` from the TA example netlist. Without them the BSIM4
drain junction current is 1e-14 A at every width and temperature. The TA netlist leaves them out.
`06_gates_spice.py diffusion` simulates every gate case with and without them
(`results/diffusion_comparison.csv`): below 50 C they differ by 0.03 % (median) and at most 0.97 %,
above 100 C by 0.9 % (median) and up to 27.6 %.

```sh
python3 scripts/10_trend_sweeps.py         # one variable at a time: OFF current vs T and vth0, Ig vs toxe
python3 scripts/10_trend_sweeps.py width   # W alone, checks the effective-width scaling (results/width_sweep.csv)
```

The sweeps hold one transistor (W = 4 Lmin) at a fixed bias with everything else nominal, because
in the random dataset the vth0 trend is buried under the spread of Vd, Vs and W.

### Transistor models

```sh
python3 scripts/03_check_data.py   # Kirchhoff check and trend plots
# the comparison of the brief, into data/comparison/ (committed as results/comparison/)
nix develop .#gpu -c python3 scripts/04_train.py --output data/comparison
# the frozen method (config/stage4_frozen.toml), into results/ and models/
nix develop .#gpu -c python3 scripts/04_train.py --models mlp --transforms log --no-curve --seed 463 \
    --separate --mask-off-ig --ensemble 5
```

`04_train.py` splits each set of each case 70/15/15 with a fixed seed. Without options it compares
polynomial ridge regression, extra trees, LightGBM and an MLP (PyTorch, GPU) on two target transforms,
`log10|I|` with a sign classifier and `asinh(I / I0)`, and keeps the model with the lowest validation
median relative error per case, averaged over the uniform and targeted sets. That run writes
`results/comparison/` (`model_comparison.csv`, `chosen_models.csv`, `learning_curve.csv` and the
figures). The frozen method is the MLP with the log target and three options: `--separate` fits a
second MLP for the gate and body currents of the ON cases, `--mask-off-ig` leaves OFF gate currents
at or below 1e-26 A out of the loss, and `--ensemble 5` averages log10|I| over seeds 463 to 467. Its
test errors are `results/chosen_models.csv`, its parity plots `results/figures/parity_*`.

How the frozen method was picked: `config/stage3_arms.toml` fixed the arms, their seeds and their kill
conditions before they ran. `scripts/run_arms.sh` trains them and `scripts/11_compare_runs.py` scores
them on the Part A validation rows only (`results/arms/`: `noise_floor.csv`, `arm_report.csv`, the
metrics of every run and `crashed_runs.csv`). Part B and C were never used to train or pick a model.

Other scripts get currents from `scripts/fet_model.py`, which needs no PyTorch:

```python
from fet_model import load_models
models = load_models(MODELS_DIR)
currents = models.predict("nmos_off", frame)   # columns id, ig, is, ib in A, positive into the terminal
```

## Part B: gates

```sh
python3 scripts/05_corners.py          # data/corners.csv, the 200 corners shared by Parts B and C
python3 scripts/06_gates_spice.py      # SPICE reference, 18 gate and input cases x 200 corners
python3 scripts/06_gates_spice.py rgatemod 200    # check: with and without the gate resistor
python3 scripts/06_gates_spice.py diffusion 200   # check: with and without AD, AS, PD and PS
python3 scripts/07_gates_ml.py oracle  # composition control with the SPICE transistor currents
python3 scripts/07_gates_ml.py <models_dir>
```

`07_gates_ml.py` also writes `results/transistor_r2_<name>.csv`, R² of log10|I| of every current the
composition takes from the models, and `results/gate_worst_<name>.csv`, the 20 largest gate errors
split exactly into the contribution of each predicted current.

Leakage of a gate or circuit is the total current delivered by every node held at VDD: the supply
and every input at logic 1. That equals static power / VDD. The SPICE leakage is summed from each
transistor's device-based terminal currents on those nodes (`leak_device`). The source currents
`-i(v_vdd)` plus the inputs at 1 (`leak_source`) are kept as a cross-check (`results/gate_device_vs_source.csv`). The oracle run of
`07_gates_ml.py` feeds each transistor's SPICE currents through the same row building, case split and
composition as the ML models, and must reproduce `leak_device` (`results/gate_results_oracle.csv`).

The ML gate leakage takes every OFF transistor's currents and every gate and body current from the
models, and the drain and source currents of ON transistors from Kirchhoff's current law at the
output and stack nodes (`balanced_currents` in `07_gates_ml.py`). An ON transistor in a gate has
microvolts across it and carries whatever the OFF network on its node leaks, so its channel current
is ill-conditioned as a function of its own bias. Summing every predicted terminal current as it is
(`results/gate_results_ml_terminal.csv`) underestimates the 14 gate cases whose leakage leaves VDD
through an ON PMOS by 24 to 114 % (mean bias), against 0.12 to 0.57 % median error for the same cases
with current balance.

`@m[id]` in ngspice BSIM4 is the channel current only, in the frame where PMOS values are negated and
drain and source swap when vds < 0. The drain terminal current also takes GIDL, the drain junction,
impact ionisation and the gate-to-drain tunnelling, so `gates.device_current_lines` builds it from the
right-hand-side stamps of ngspice-45 `b4ld.c`. `results/drain_current_check.csv` compares it with
the exact drain and source source currents of single transistors in all four cases and both
orientations.

The gate and adder decks set `rgatemod = 0` in the model cards (`gates.without_gate_resistor`). The
Part A data keeps the card as given. With `rgatemod = 1` the gate resistor is 0.13 ohm, so a gate
driven by another gate's output near VDD carries about 2e-15 A of rounding noise against
`abstol = 1e-18`, and several AND2 operating points never converge. Over the 18 gate cases at all
200 Part B corners (`results/rgatemod_comparison.csv`), `rgatemod = 1` failed in 61 of 3600 decks,
all AND2, and `rgatemod = 0` converged in all 3600. Two more `rgatemod = 1` decks (AND2 10 at corner
88, OR2 00 at corner 188) converged to a point whose source-based leakage is 3 and 14 times off the
device-based one, so Kirchhoff's law did not close at the supply. On the other 3537 the leakages
differ by at most 0.51 % relative (four AND2 decks above 0.1 %) and by at most 5e-5 in 99 % of them.

Gates (widths in units of Lmin = 22 nm): INV Wn = 1, Wp = 2. NAND2 Wn = 2 (series, B at the output),
Wp = 2. NOR2 Wn = 1, Wp = 4 (series, A at the supply). AND2 and OR2 are NAND2 and NOR2 followed by the
INV, simulated as complete gates.

## Part C: 3-bit carry-lookahead adder

```sh
python3 scripts/08_cla_spice.py        # SPICE reference, 128 vectors x 200 corners
python3 scripts/09_cla_ml.py oracle    # gate sums from 07_gates_ml.py oracle
python3 scripts/09_cla_ml.py ml        # gate sums from 07_gates_ml.py <models_dir>
python3 scripts/13_timing.py           # time of one adder leakage, SPICE against our method
```

The adder uses only the five gates: P_i = XOR(a_i, b_i) with XOR as four NAND2, G_i = AND2(a_i, b_i),
C1 to C3 as sums of products from AND2 and OR2 trees, S_i = XOR(P_i, C_i) and Cout = C3. That is
42 gates (24 NAND2, 12 AND2, 6 OR2) and 204 transistors. `08_cla_spice.py` first checks at the nominal corner that the outputs give a + b + cin
for all 128 vectors. `09_cla_ml.py` sums, for each vector, the Part B leakage of every gate at the
same corner and input combination. The sum of the Part B SPICE gate leakages against the SPICE adder
(`results/cla_composition_control.csv`) is the error of composing gates at all, from loading and gate
inputs that sit slightly off the rails, before any transistor model enters.

## Where the committed results come from

The first version ran on an RTX 3050 Ti, this version on an RTX 4070 Laptop GPU, with ngspice-45. The
same seed on the same GPU gives the same models, but not across the two GPUs, so every comparison is
made on one of them. `results/run_config.json` and `results/comparison/run_config.json` record the
commit each training ran from.

| Results | Commit |
| --- | --- |
| Gate and adder SPICE (`06`, `08`): `gate_device_vs_source`, `cla_device_vs_source`, `cla_logic_check`, `drain_current_check` | 88997d3 (first version) |
| One-variable sweeps (`10`): `trend_sweeps`, `trend_sweep` | 1499b31 (first version) |
| Arms of stage 2 and 3 (`run_arms.sh`, `11`): `results/arms/` | 79d5761, committed in 516bc9c |
| Blind evaluation (`12`) on the stage 4 corners: `results/stage4/` | eea82ab on `accuracy-stage4`, committed in 30b2c05 |
| Frozen models and their results (`04`, `07`, `09`): `models/`, `chosen_models`, `parity_*`, `gate_results_*`, `cla_results_*`, `cla_composition_control`, `cla_net_logic_check` | 0826d56, committed in 83b5f72 |
| Model comparison and learning curve: `results/comparison/` | 638a0f0. `model_comparison.csv` and `chosen_models.csv` were rerun with `--no-curve` to add `r2_log10` (the commit after ad1a69e). Every other column came out identical |
| Checks: `rgatemod_comparison`, `diffusion_comparison`, `transistor_r2_*`, `gate_worst_*`, `width_sweep` | committed in e4cbbb5 and 4cb4667 |
| Kirchhoff check with the gate-bias set (`03`), error analysis and model tables: `kirchhoff`, `report/analysis/` | the commit that holds them |
