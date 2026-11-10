"""Train and compare four model types for each single-transistor case, and keep the best per case.

Usage: python3 scripts/04_train.py [--models poly_ridge,extra_trees,lightgbm,mlp] [--transforms log,asinh]
                                   [--cases nmos_on,...] [--seed 463] [--training-sets uniform,targeted]
                                   [--select median | gatebias] [--no-curve] [--output DIR]
                                   [--weff] [--separate] [--mask-off-ig] [--ensemble 5]

Reads data/fet_<case>.csv (uniform set), data/fet_<case>_targeted.csv (drain and source near each
other or at the rails) and data/fet_<case>_gatebias.csv (the bias inside a static gate), see
01_sample_fet.py. Fits on the training parts of --training-sets and evaluates on every set. Writes
<output>/results/model_comparison.csv (validation), model_scores.csv (validation, one row per model),
chosen_models.csv (test, chosen model only), learning_curve.csv (validation), run_config.json, the
parity plots results/figures/parity_<case>.{pdf,png}, results/figures/learning_curve.{pdf,png}, and
the chosen models as <output>/models/fet_<case>.joblib, which scripts/fet_model.py loads. Without
--output, <output> is the repository, so the results land in results/ and models/.

The defaults reproduce the run behind the committed models: all four model types, both transforms,
seed 463, training on the uniform and targeted sets, selection by the median score. --training-sets
uniform is the uniform-only run.

Each set of each case is split 70/15/15 into training, validation and test rows with a fixed seed,
and the models fit on the training parts of the training sets together. Every error is reported for
each set apart. Every model predicts all four terminal currents of its case from the features of
fet_model.features. Two target transforms are compared:
- log: log10|I| per terminal, with the sign taken from a classifier of the same model type for the
  terminals whose sign changes within the training set, and the constant sign otherwise.
- asinh: asinh(I / I0), with I0 the 1st percentile of the nonzero |I| in the training set per case and terminal.
  asinh is linear for |I| below I0 and logarithmic above it, so 99 % of the training currents fall in
  the logarithmic part, where an error in the target is a relative error in the current, and the
  sign change of Id and Is passes smoothly through zero.

Metrics per case and terminal: R2 of asinh(I / I0) (the same I0 for every model, so the two transforms
compare), and the mean, median and 95th percentile of |predicted - SPICE| / |SPICE|. Two selection
scores per model, both on validation rows:
- median: the median relative error, averaged over the four terminals and the training sets. This is
  the score behind the committed models.
- gatebias: on the gate-bias set, sum |predicted - SPICE| / sum |SPICE| over the currents the
  Kirchhoff composition of 07_gates_ml.py uses. That is Ig and Ib for ON cases, whose drain and source
  currents come from current balance, and all four for OFF cases (see gate_bias_score).
--select picks which one chooses the model. The learning curve takes the first rows of the training
parts in proportion.

Method options, each off by default (see fit_case_model):
--weff         ON cases predict Ig and Ib per effective width W - 2 wint (wint from the model file), the
               gate and body currents scale with it, and get log_weff as a feature
--separate     ON cases fit a second regressor for Ig and Ib alone
--mask-off-ig  OFF cases leave Ig at or below 1e-26 A out of the MLP loss
--ensemble N   the chosen model plus N - 1 more seeds (seed + 1 onwards), averaged in log10|I|
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import CASES, DATA_DIR, DEFAULT_TRAINING_SETS, ROOT, SAMPLE_SETS, TERMINALS, channel_width_offsets
from fet_model import CURRENT_FLOOR, features, predict_bundle, to_target, width_divisor

SPLIT_SEED = 463
DEFAULT_MODEL_SEED = 463
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
MODEL_TYPES = ["poly_ridge", "extra_trees", "lightgbm", "mlp"]
TRANSFORMS = ["log", "asinh"]
SELECTION_SCORES = ["median", "gatebias"]
LEARNING_CURVE_SIZES = [1000, 2000, 5000, 7000]
CURRENT_SCALE_PERCENTILE = 1.0
# The training run leaves the other cores to SPICE jobs running at the same time.
CPU_WORKERS = 8
# OFF gate currents at or below this are far under any leakage the gates add up, and ngspice gives
# some of them as exactly zero, so --mask-off-ig leaves them out of the loss.
GATEBIAS_OFF_IG_FLOOR = 1e-26

POLY_DEGREE = 3
RIDGE_ALPHAS = np.logspace(-8, 2, 11)
TREE_COUNT = 300
BOOSTING_ROUNDS = 2000
BOOSTING_LEARNING_RATE = 0.03
MLP_WIDTH = 256
MLP_DEPTH = 4
MLP_EPOCHS = 2000
MLP_BATCH = 512
MLP_LEARNING_RATE = 2e-3

COLORS = {"nmos": "#2a6fb0", "pmos": "#c0582b"}
TERMINAL_LABELS = {"id": "Id", "ig": "Ig", "is": "Is", "ib": "Ib"}


def split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = np.random.default_rng(SPLIT_SEED).permutation(len(data))
    train_end = int(round(TRAIN_FRACTION * len(data)))
    validation_end = train_end + int(round(VALIDATION_FRACTION * len(data)))
    return (data.iloc[order[:train_end]], data.iloc[order[train_end:validation_end]],
            data.iloc[order[validation_end:]])


def train_mlp(inputs: np.ndarray, targets: np.ndarray, loss_name: str, seed: int,
              loss_weights: np.ndarray | None = None) -> dict:
    """Fully connected SiLU network trained with Adam on the GPU, returned as numpy weights.

    loss_weights (rows, outputs) of 0 or 1 leaves entries out of the mean squared error.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the MLP trains on the GPU and torch.cuda.is_available() is False")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    layers = []
    width_in = inputs.shape[1]
    for _ in range(MLP_DEPTH):
        layers += [torch.nn.Linear(width_in, MLP_WIDTH), torch.nn.SiLU()]
        width_in = MLP_WIDTH
    layers.append(torch.nn.Linear(width_in, targets.shape[1]))
    network = torch.nn.Sequential(*layers).to(device)
    inputs_gpu = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    targets_gpu = torch.as_tensor(targets, dtype=torch.float32, device=device)
    if loss_weights is None:
        loss_function = torch.nn.MSELoss() if loss_name == "mse" else torch.nn.BCEWithLogitsLoss()
    else:
        if loss_name != "mse":
            raise ValueError("loss weights apply to the regression loss only")
        weights_gpu = torch.as_tensor(loss_weights, dtype=torch.float32, device=device)
        state = {"batch": None}

        def loss_function(predicted, target):
            # The batch rows are set before each call, the full set when state["batch"] is None.
            weights = weights_gpu if state["batch"] is None else weights_gpu[state["batch"]]
            return ((predicted - target) ** 2 * weights).sum() / weights.sum().clamp(min=1.0)
    sample_count = len(inputs)
    steps_per_epoch = -(-sample_count // MLP_BATCH)
    optimizer = torch.optim.Adam(network.parameters(), lr=MLP_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MLP_LEARNING_RATE,
                                                    total_steps=MLP_EPOCHS * steps_per_epoch)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(MLP_EPOCHS):
        order = torch.randperm(sample_count, device=device, generator=generator)
        for start in range(0, sample_count, MLP_BATCH):
            batch = order[start:start + MLP_BATCH]
            if loss_weights is not None:
                state["batch"] = batch
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(inputs_gpu[batch]), targets_gpu[batch])
            loss.backward()
            optimizer.step()
            scheduler.step()
    if loss_weights is not None:
        state["batch"] = None
    with torch.no_grad():
        final_loss = loss_function(network(inputs_gpu), targets_gpu).item()
    linear_layers = [layer for layer in network if isinstance(layer, torch.nn.Linear)]
    return {
        "kind": "mlp",
        "weights": [layer.weight.detach().cpu().numpy().T.astype(np.float64) for layer in linear_layers],
        "biases": [layer.bias.detach().cpu().numpy().astype(np.float64) for layer in linear_layers],
        "final_training_loss": final_loss,
    }


def fit_regressor(model_type: str, inputs: np.ndarray, targets: np.ndarray, seed: int,
                  loss_weights: np.ndarray | None = None) -> dict:
    if model_type == "mlp":
        return train_mlp(inputs, targets, "mse", seed, loss_weights)
    if loss_weights is not None:
        raise ValueError(f"loss weights are implemented for the MLP only, not {model_type}")
    if model_type == "poly_ridge":
        estimator = make_pipeline(PolynomialFeatures(POLY_DEGREE, include_bias=False), StandardScaler(),
                                  RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True))
    elif model_type == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=TREE_COUNT, n_jobs=CPU_WORKERS, random_state=seed)
    elif model_type == "lightgbm":
        estimator = MultiOutputRegressor(LGBMRegressor(
            n_estimators=BOOSTING_ROUNDS, learning_rate=BOOSTING_LEARNING_RATE, num_leaves=31,
            min_child_samples=10, n_jobs=CPU_WORKERS, random_state=seed, verbose=-1))
    else:
        raise ValueError(f"unknown model type {model_type}")
    estimator.fit(inputs, targets)
    return {"kind": "sklearn", "estimator": estimator}


def fit_sign_classifier(model_type: str, inputs: np.ndarray, positive: np.ndarray, seed: int) -> dict:
    """One binary output per terminal whose sign changes in the training set."""
    if model_type == "mlp":
        return train_mlp(inputs, positive.astype(np.float64), "bce", seed)
    if model_type == "poly_ridge":
        estimator = MultiOutputClassifier(make_pipeline(
            PolynomialFeatures(POLY_DEGREE, include_bias=False), StandardScaler(),
            LogisticRegression(max_iter=5000)))
    elif model_type == "extra_trees":
        estimator = ExtraTreesClassifier(n_estimators=TREE_COUNT, n_jobs=CPU_WORKERS, random_state=seed)
    elif model_type == "lightgbm":
        estimator = MultiOutputClassifier(LGBMClassifier(
            n_estimators=BOOSTING_ROUNDS // 4, learning_rate=BOOSTING_LEARNING_RATE, num_leaves=31,
            min_child_samples=10, n_jobs=CPU_WORKERS, random_state=seed, verbose=-1))
    else:
        raise ValueError(f"unknown model type {model_type}")
    estimator.fit(inputs, positive)
    return {"kind": "sklearn", "estimator": estimator}


def fit_case_model(model_type: str, transform: str, train: pd.DataFrame, case_name: str,
                   current_scale: np.ndarray, seed: int, options: dict) -> dict:
    """Fit one model of one case and return it as the bundle fet_model.predict_bundle reads.

    options (each False by default, all from the command line):
    - weff: ON cases predict Ig and Ib per effective width W - 2 * wint, with log_weff as a feature
    - separate: ON cases fit a second regressor for Ig and Ib alone
    - mask_off_ig: OFF cases leave Ig at or below GATEBIAS_OFF_IG_FLOOR out of the MLP loss
    """
    device = CASES[case_name]["device"]
    is_on = case_name.endswith("_on")
    gate_and_body = [TERMINALS.index("ig"), TERMINALS.index("ib")]
    width_offset = channel_width_offsets()[device] if options["weff"] and is_on else None
    scaled_terminals = gate_and_body if width_offset is not None else []
    raw_inputs = features(train, device, width_offset)
    feature_mean = raw_inputs.mean(axis=0)
    feature_std = raw_inputs.std(axis=0)
    if np.any(feature_std == 0):
        raise ValueError(f"{case_name}: a feature is constant over the training rows")
    inputs = (raw_inputs - feature_mean) / feature_std
    currents = train[TERMINALS].to_numpy()
    divisor = width_divisor(train, width_offset, scaled_terminals) if scaled_terminals else None
    targets = to_target(currents, transform, current_scale, divisor)
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0)
    normalized = (targets - target_mean) / target_std
    loss_weights = None
    if options["mask_off_ig"] and not is_on:
        loss_weights = np.ones_like(normalized)
        gate_column = TERMINALS.index("ig")
        loss_weights[:, gate_column] = np.abs(currents[:, gate_column]) > GATEBIAS_OFF_IG_FLOOR
    regressor = fit_regressor(model_type, inputs, normalized, seed, loss_weights)
    separate = None
    if options["separate"] and is_on:
        separate = {"terminals": gate_and_body,
                    "regressor": fit_regressor(model_type, inputs, normalized[:, gate_and_body], seed),
                    "target_mean": target_mean[gate_and_body], "target_std": target_std[gate_and_body]}

    constant_signs = np.zeros(len(TERMINALS))
    mixed_terminals = []
    for index in range(len(TERMINALS)):
        if np.all(currents[:, index] > 0):
            constant_signs[index] = 1.0
        elif np.all(currents[:, index] < 0):
            constant_signs[index] = -1.0
        else:
            mixed_terminals.append(index)
    sign_classifier = None
    if transform == "log" and mixed_terminals:
        sign_classifier = fit_sign_classifier(model_type, inputs, currents[:, mixed_terminals] > 0, seed)
    return {
        "case": case_name,
        "device": device,
        "model_type": model_type,
        "transform": transform,
        "seed": seed,
        "current_scale": current_scale,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "regressor": regressor,
        "separate": separate,
        "width_offset": width_offset,
        "width_scaled_terminals": scaled_terminals,
        "options": dict(options),
        "constant_signs": constant_signs,
        "mixed_terminals": mixed_terminals,
        "sign_classifier": sign_classifier,
        "training_rows": len(train),
    }


def relative_errors(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(predicted - true) / np.maximum(np.abs(true), CURRENT_FLOOR)


def error_rows(true: np.ndarray, predicted: np.ndarray, current_scale: np.ndarray) -> list[dict]:
    relative = relative_errors(true, predicted)
    rows = []
    for index, terminal in enumerate(TERMINALS):
        true_target = np.arcsinh(true[:, index] / current_scale[index])
        predicted_target = np.arcsinh(predicted[:, index] / current_scale[index])
        residual = np.sum((predicted_target - true_target) ** 2)
        spread = np.sum((true_target - true_target.mean()) ** 2)
        # The same on log10|I| with the log target's floor: the magnitude over all decades, sign
        # errors are counted apart in sign_error_fraction.
        true_log = np.log10(np.maximum(np.abs(true[:, index]), CURRENT_FLOOR))
        predicted_log = np.log10(np.maximum(np.abs(predicted[:, index]), CURRENT_FLOOR))
        log_residual = np.sum((predicted_log - true_log) ** 2)
        log_spread = np.sum((true_log - true_log.mean()) ** 2)
        rows.append({
            "terminal": terminal,
            "r2_asinh": 1.0 - residual / spread,
            "r2_log10": 1.0 - log_residual / log_spread,
            "mean_rel_error": np.mean(relative[:, index]),
            "median_rel_error": np.median(relative[:, index]),
            "p95_rel_error": np.percentile(relative[:, index], 95),
            "sign_error_fraction": np.mean(np.sign(predicted[:, index]) != np.sign(true[:, index])),
        })
    return rows


def gate_bias_score(case_name: str, true: np.ndarray, predicted: np.ndarray) -> float:
    """Size-weighted error of the currents the Kirchhoff composition uses: sum |predicted - SPICE| / sum |SPICE|.

    ON cases use Ig and Ib, OFF cases all four currents. A gate's leakage is a sum of such currents, so
    each current counts by its size, as it does there. A plain mean of relative errors was tried first
    and was ruled by OFF gate currents of 1e-26 to 1e-19 A, seven decades below any gate leakage: on
    the committed models their mean relative error was 142 and the OFF score 35, against 0.095 weighted.
    """
    columns = [TERMINALS.index("ig"), TERMINALS.index("ib")] if case_name.endswith("_on") else list(range(len(TERMINALS)))
    return float(np.abs(predicted[:, columns] - true[:, columns]).sum() / np.abs(true[:, columns]).sum())


def median_score(rows: list[dict]) -> float:
    return float(np.mean([row["median_rel_error"] for row in rows]))


def evaluate(bundle: dict, parts: dict, part_index: int, current_scale: np.ndarray) -> tuple[dict, dict]:
    """Error rows and predictions of one model on one part (1 validation, 2 test) of every set."""
    rows, predictions = {}, {}
    for set_name, split_parts in parts.items():
        frame = split_parts[part_index]
        predicted = predict_bundle(bundle, frame).to_numpy()
        rows[set_name] = error_rows(frame[TERMINALS].to_numpy(), predicted, current_scale)
        predictions[set_name] = predicted
    return rows, predictions


def scores(case_name: str, rows: dict, predictions: dict, parts: dict, part_index: int,
           training_sets: list[str]) -> dict:
    true = parts["gatebias"][part_index][TERMINALS].to_numpy()
    return {
        "median": float(np.mean([median_score(rows[set_name]) for set_name in training_sets])),
        "gatebias": gate_bias_score(case_name, true, predictions["gatebias"]),
    }


def save_figure(figure, figure_dir: Path, name: str, rect: tuple = (0, 0, 1, 1)) -> None:
    """rect is the part of the figure tight_layout fills, smaller to leave room for a figure legend."""
    figure.tight_layout(rect=rect)
    for suffix in ("pdf", "png"):
        figure.savefig(figure_dir / f"{name}.{suffix}", dpi=200)
    plt.close(figure)


def parity_plot(case_name: str, sets: dict, figure_dir: Path) -> None:
    """sets maps a set name to its (SPICE, predicted) test currents."""
    device = CASES[case_name]["device"]
    other_device = "pmos" if device == "nmos" else "nmos"
    set_colors = {"uniform": COLORS[device], "targeted": "0.35", "gatebias": "#1baf7a"}
    figure, axes = plt.subplots(1, len(TERMINALS), figsize=(11, 3.1))
    for index, (axis, terminal) in enumerate(zip(axes, TERMINALS)):
        magnitudes = [np.maximum(np.abs(currents[:, index]), CURRENT_FLOOR)
                      for pair in sets.values() for currents in pair]
        low = min(values.min() for values in magnitudes) / 2
        high = max(values.max() for values in magnitudes) * 2
        axis.plot([low, high], [low, high], color="0.6", linewidth=1)
        for set_name, (true, predicted) in sets.items():
            spice = np.maximum(np.abs(true[:, index]), CURRENT_FLOOR)
            model = np.maximum(np.abs(predicted[:, index]), CURRENT_FLOOR)
            wrong_sign = np.sign(predicted[:, index]) != np.sign(true[:, index])
            axis.scatter(spice[~wrong_sign], model[~wrong_sign], s=3, alpha=0.5, color=set_colors[set_name],
                         rasterized=True)
            # Wrong signs of every set share one colour, the other device's, so they stand out.
            axis.scatter(spice[wrong_sign], model[wrong_sign], s=6, color=COLORS[other_device], rasterized=True)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_title(TERMINAL_LABELS[terminal])
        axis.set_xlabel("SPICE |I| (A)")
        if index == 0:
            axis.set_ylabel("ML |I| (A)")
    # One legend for the four panels, above them, so that it covers no data.
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=set_colors[set_name], label=set_name)
               for set_name in sets]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=COLORS[other_device],
                              label="sign wrong"))
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8)
    save_figure(figure, figure_dir, f"parity_{case_name}", rect=(0, 0, 1, 0.92))


def learning_curve_plot(curve: pd.DataFrame, figure_dir: Path) -> None:
    """One row of panels per set: median and 95th percentile error against the training rows."""
    set_count = curve["set"].nunique()
    figure, axes = plt.subplots(set_count, 2, figsize=(7, 2.8 * set_count), sharex=True, squeeze=False)
    for row_axes, (set_name, set_rows) in zip(axes, curve.groupby("set", sort=False)):
        for case_name, rows in set_rows.groupby("case", sort=False):
            summary = rows.groupby("train_size")[["median_rel_error", "p95_rel_error"]].mean()
            style = {"color": COLORS[CASES[case_name]["device"]],
                     "linestyle": "-" if case_name.endswith("_on") else "--", "marker": "o", "markersize": 3}
            row_axes[0].plot(summary.index, summary["median_rel_error"], label=case_name, **style)
            row_axes[1].plot(summary.index, summary["p95_rel_error"], label=case_name, **style)
        for axis, label in zip(row_axes, ["Median relative error", "95th pct relative error"]):
            axis.set_yscale("log")
            axis.set_ylabel(label)
            axis.set_title(f"{set_name} validation rows", fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("Training rows")
    axes[0][0].legend(frameon=False, fontsize=7)
    save_figure(figure, figure_dir, "learning_curve")


def score_text(case_scores: dict) -> str:
    return "  ".join(f"{name} {value:.4f}" for name, value in case_scores.items())


def train_case(case_name: str, model_types: list[str], transforms: list[str], seed: int,
               training_sets: list[str], select: str, learning_curve: bool, options: dict, ensemble_size: int,
               models_dir: Path, figure_dir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    # Each set name maps to its (train, validation, test) parts.
    parts = {}
    for set_name, suffix in SAMPLE_SETS.items():
        path = DATA_DIR / f"fet_{case_name}{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing, run scripts/01_sample_fet.py and 02_sim_fet.py")
        parts[set_name] = split(pd.read_csv(path))
    for set_name, (train_part, validation_part, test_part) in parts.items():
        role = "training" if set_name in training_sets else "evaluation only"
        print(f"{case_name} {set_name} ({role}): {len(train_part)} train, {len(validation_part)} validation, "
              f"{len(test_part)} test rows", flush=True)
    train = pd.concat([parts[set_name][0] for set_name in training_sets], ignore_index=True)
    train_currents = train[TERMINALS].to_numpy()
    # Over the nonzero currents: an OFF device with drain and source near 0 V can have a gate current of
    # exactly zero in ngspice, and a zero I0 would make asinh(I / I0) undefined.
    current_scale = np.array([np.percentile(np.abs(column[column != 0]), CURRENT_SCALE_PERCENTILE)
                              for column in train_currents.T])
    print("  I0: " + ", ".join(f"{terminal} {scale:.2e} A" for terminal, scale in zip(TERMINALS, current_scale)),
          flush=True)

    comparison, model_scores = [], []
    # Null arm: the training median of each terminal current, whatever the inputs.
    null_median = np.median(train_currents, axis=0)
    null_rows, null_predictions = {}, {}
    for set_name, (_, validation_part, _) in parts.items():
        null_predictions[set_name] = np.tile(null_median, (len(validation_part), 1))
        null_rows[set_name] = error_rows(validation_part[TERMINALS].to_numpy(), null_predictions[set_name],
                                         current_scale)
        for row in null_rows[set_name]:
            comparison.append({"case": case_name, "set": set_name, "model": "null_median", "transform": "none",
                               "seed": seed, **row, "fit_seconds": 0.0})
    null_scores = scores(case_name, null_rows, null_predictions, parts, 1, training_sets)
    print(f"  {'null_median':12s} {'none':6s} {score_text(null_scores)}", flush=True)
    model_scores.append({"case": case_name, "model": "null_median", "transform": "none", "seed": seed,
                         **{f"{name}_score": value for name, value in null_scores.items()}, "fit_seconds": 0.0})

    best = None
    for model_type in model_types:
        for transform in transforms:
            start = time.time()
            bundle = fit_case_model(model_type, transform, train, case_name, current_scale, seed, options)
            fit_seconds = time.time() - start
            rows, predictions = evaluate(bundle, parts, 1, current_scale)
            case_scores = scores(case_name, rows, predictions, parts, 1, training_sets)
            print(f"  {model_type:12s} {transform:6s} {score_text(case_scores)}  ({fit_seconds:.0f} s)", flush=True)
            for set_name, set_rows in rows.items():
                for row in set_rows:
                    comparison.append({"case": case_name, "set": set_name, "model": model_type,
                                       "transform": transform, "seed": seed, **row, "fit_seconds": fit_seconds})
            model_scores.append({"case": case_name, "model": model_type, "transform": transform, "seed": seed,
                                 **{f"{name}_score": value for name, value in case_scores.items()},
                                 "fit_seconds": fit_seconds})
            if best is None or case_scores[select] < best[0][select]:
                best = (case_scores, bundle)
    validation_scores, chosen = best
    print(f"  chosen by {select}: {chosen['model_type']} {chosen['transform']}", flush=True)
    if ensemble_size > 1:
        # The chosen model and the next seeds in order, so no seed is picked by its score.
        start = time.time()
        members = [chosen] + [fit_case_model(chosen["model_type"], chosen["transform"], train, case_name,
                                             current_scale, seed + offset, options)
                              for offset in range(1, ensemble_size)]
        fit_seconds = time.time() - start
        chosen = {"kind": "ensemble", "members": members, "case": case_name, "device": chosen["device"],
                  "model_type": f"{chosen['model_type']}_ensemble{ensemble_size}", "transform": chosen["transform"],
                  "seed": seed}
        rows, predictions = evaluate(chosen, parts, 1, current_scale)
        validation_scores = scores(case_name, rows, predictions, parts, 1, training_sets)
        print(f"  {chosen['model_type']:12s} {chosen['transform']:6s} {score_text(validation_scores)}  "
              f"({fit_seconds:.0f} s)", flush=True)
        for set_name, set_rows in rows.items():
            for row in set_rows:
                comparison.append({"case": case_name, "set": set_name, "model": chosen["model_type"],
                                   "transform": chosen["transform"], "seed": seed, **row, "fit_seconds": fit_seconds})
        model_scores.append({"case": case_name, "model": chosen["model_type"], "transform": chosen["transform"],
                             "seed": seed, **{f"{name}_score": value for name, value in validation_scores.items()},
                             "fit_seconds": fit_seconds})
    joblib.dump(chosen, models_dir / f"fet_{case_name}.joblib", compress=3)

    test_rows, test_predictions = evaluate(chosen, parts, 2, current_scale)
    test_scores = scores(case_name, test_rows, test_predictions, parts, 2, training_sets)
    chosen_rows = []
    for set_name, set_rows in test_rows.items():
        chosen_rows += [{"case": case_name, "set": set_name, "model": chosen["model_type"],
                         "transform": chosen["transform"], "seed": seed, "selected_by": select,
                         "validation_median_score": validation_scores["median"],
                         "validation_gatebias_score": validation_scores["gatebias"],
                         "test_gatebias_score": test_scores["gatebias"], **row} for row in set_rows]
    parity_plot(case_name, {set_name: (parts[set_name][2][TERMINALS].to_numpy(), test_predictions[set_name])
                            for set_name in parts}, figure_dir)

    curve = []
    if not learning_curve:
        return comparison, model_scores, chosen_rows, curve
    reference_rows = len(parts[training_sets[0]][0])
    for size in LEARNING_CURVE_SIZES:
        if size > reference_rows:
            print(f"  learning curve: skip {size} rows, the {training_sets[0]} training part has {reference_rows}",
                  flush=True)
            continue
        # The same fraction of each training part, so the mix of the sets stays as in the full fit.
        subset = pd.concat([parts[set_name][0].iloc[:int(round(size / reference_rows * len(parts[set_name][0])))]
                            for set_name in training_sets], ignore_index=True)
        bundle = fit_case_model(chosen["model_type"], chosen["transform"], subset, case_name, current_scale, seed,
                                options)
        rows, predictions = evaluate(bundle, parts, 1, current_scale)
        print(f"  learning curve {len(subset):5d} rows: "
              f"{score_text(scores(case_name, rows, predictions, parts, 1, training_sets))}", flush=True)
        for set_name, set_rows in rows.items():
            for row in set_rows:
                curve.append({"case": case_name, "set": set_name, "model": chosen["model_type"],
                              "transform": chosen["transform"], "seed": seed, "train_size": len(subset), **row})
    return comparison, model_scores, chosen_rows, curve


def subset_argument(text: str, allowed: list[str], name: str) -> list[str]:
    values = text.split(",")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {name} {unknown}, choose from {allowed}")
    return values


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                            text=True, check=True)
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(MODEL_TYPES), help="subset of " + ",".join(MODEL_TYPES))
    parser.add_argument("--transforms", default=",".join(TRANSFORMS), help="subset of " + ",".join(TRANSFORMS))
    parser.add_argument("--cases", default=",".join(CASES), help="subset of " + ",".join(CASES))
    parser.add_argument("--seed", type=int, default=DEFAULT_MODEL_SEED, help="model seed, the split seed stays fixed")
    parser.add_argument("--training-sets", default=",".join(DEFAULT_TRAINING_SETS),
                        help="sets to fit on, subset of " + ",".join(SAMPLE_SETS))
    parser.add_argument("--select", default="median", choices=SELECTION_SCORES,
                        help="validation score that chooses the model of each case")
    parser.add_argument("--no-curve", action="store_true", help="skip the learning curve")
    parser.add_argument("--weff", action="store_true",
                        help="ON cases: Ig and Ib per effective width W - 2 wint, log_weff as a feature")
    parser.add_argument("--separate", action="store_true", help="ON cases: a second regressor for Ig and Ib")
    parser.add_argument("--mask-off-ig", action="store_true",
                        help=f"OFF cases: Ig at or below {GATEBIAS_OFF_IG_FLOOR:g} A out of the MLP loss")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="members per case, the chosen model and the next seeds, averaged in log10|I|")
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="directory that gets models/ and results/, the repository by default")
    arguments = parser.parse_args()
    model_types = subset_argument(arguments.models, MODEL_TYPES, "model types")
    transforms = subset_argument(arguments.transforms, TRANSFORMS, "transforms")
    case_names = subset_argument(arguments.cases, list(CASES), "cases")
    training_sets = subset_argument(arguments.training_sets, list(SAMPLE_SETS), "training sets")
    options = {"weff": arguments.weff, "separate": arguments.separate, "mask_off_ig": arguments.mask_off_ig}
    if arguments.ensemble < 1 or arguments.ensemble % 2 == 0:
        raise ValueError("--ensemble must be odd, so the sign vote of the members cannot tie")
    if arguments.ensemble > 1 and not arguments.no_curve:
        raise ValueError("--ensemble needs --no-curve, the learning curve fits single models")
    if (arguments.weff or arguments.mask_off_ig) and transforms != ["log"]:
        raise ValueError("--weff and --mask-off-ig are defined for the log transform only, add --transforms log")
    if arguments.mask_off_ig and model_types != ["mlp"]:
        raise ValueError("--mask-off-ig is implemented for the MLP only, add --models mlp")
    if "mlp" in model_types:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("the MLP trains on the GPU and torch.cuda.is_available() is False")
        print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)

    models_dir = arguments.output / "models"
    results_dir = arguments.output / "results"
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    config = {"models": model_types, "transforms": transforms, "cases": case_names, "seed": arguments.seed,
              "training_sets": training_sets, "select": arguments.select, "learning_curve": not arguments.no_curve,
              "options": options, "ensemble": arguments.ensemble, "git": git_state()}
    (results_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)

    comparison, model_scores, chosen, curve = [], [], [], []
    for case_name in case_names:
        case_results = train_case(case_name, model_types, transforms, arguments.seed, training_sets,
                                  arguments.select, not arguments.no_curve, options, arguments.ensemble,
                                  models_dir, figure_dir)
        for collected, part in zip((comparison, model_scores, chosen, curve), case_results):
            collected += part

    pd.DataFrame(comparison).to_csv(results_dir / "model_comparison.csv", index=False, float_format="%.6g")
    pd.DataFrame(model_scores).to_csv(results_dir / "model_scores.csv", index=False, float_format="%.6g")
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(results_dir / "chosen_models.csv", index=False, float_format="%.6g")
    if curve:
        curve = pd.DataFrame(curve)
        curve.to_csv(results_dir / "learning_curve.csv", index=False, float_format="%.6g")
        learning_curve_plot(curve, figure_dir)
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    main()
