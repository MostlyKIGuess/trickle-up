"""Metrics of training runs on Part A validation rows, and the arm table of config/stage3_arms.toml.

Usage: python3 scripts/11_compare_runs.py metrics <run_dir> [<run_dir> ...]
       python3 scripts/11_compare_runs.py report <runs_root>

A run directory is the --output of 04_train.py, with models/ and results/. Its name is <label>_s<seed>,
for example baseline_s463 or A1_s464. "metrics" writes <run_dir>/results/arm_metrics.csv, one row per
case plus a row "all". "report" reads every arm_metrics.csv under <runs_root>, prints and writes
<runs_root>/noise_floor.csv and <runs_root>/arm_report.csv with the kill verdicts of the config.

Only validation rows of the Part A sets are used (the split of 04_train.py), never Part B or Part C.
Every error is a mean relative error |predicted - SPICE| / |SPICE|. The metrics are described in
config/stage3_arms.toml.
"""

import importlib
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from common import CASES, DATA_DIR, L_MIN, ROOT, SAMPLE_SETS, TERMINALS
from fet_model import load_models

train_module = importlib.import_module("04_train")

CONFIG_PATH = ROOT / "config" / "stage3_arms.toml"
NARROW_WIDTH = 1.5 * L_MIN
WIDE_WIDTH = 2.0 * L_MIN
HOT_TEMPERATURE = 92.0
METRICS = ["gatebias_score", "median_score", "on_ig", "on_ig_narrow_ratio", "off_id_across", "hot_score"]


def validation_part(case_name: str, set_name: str) -> pd.DataFrame:
    return train_module.split(pd.read_csv(DATA_DIR / f"fet_{case_name}{SAMPLE_SETS[set_name]}.csv"))[1]


def mean_relative(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(train_module.relative_errors(true, predicted).mean())


def run_metrics(run_dir: Path) -> pd.DataFrame:
    models = load_models(run_dir / "models")
    gate_column = TERMINALS.index("ig")
    drain_column = TERMINALS.index("id")
    rows = []
    for case_name in CASES:
        row = {"case": case_name}
        gatebias = validation_part(case_name, "gatebias")
        true = gatebias[TERMINALS].to_numpy()
        predicted = models.predict(case_name, gatebias).to_numpy()
        row["gatebias_score"] = train_module.gate_bias_score(case_name, true, predicted)
        hot = (gatebias["temp"] > HOT_TEMPERATURE).to_numpy()
        row["hot_score"] = train_module.gate_bias_score(case_name, true[hot], predicted[hot])
        median_scores = []
        for set_name in ("uniform", "targeted"):
            frame = validation_part(case_name, set_name)
            set_predicted = models.predict(case_name, frame).to_numpy()
            median_scores.append(np.mean(np.median(
                train_module.relative_errors(frame[TERMINALS].to_numpy(), set_predicted), axis=0)))
        row["median_score"] = float(np.mean(median_scores))
        if case_name.endswith("_on"):
            row["on_ig"] = mean_relative(true[:, gate_column], predicted[:, gate_column])
            narrow = (gatebias["w"] < NARROW_WIDTH).to_numpy()
            wide = (gatebias["w"] > WIDE_WIDTH).to_numpy()
            row["on_ig_narrow_ratio"] = (mean_relative(true[narrow, gate_column], predicted[narrow, gate_column])
                                         / mean_relative(true[wide, gate_column], predicted[wide, gate_column]))
        else:
            across = (np.abs(gatebias["vd"] - gatebias["vs"]) > gatebias["vdd"] / 2).to_numpy()
            row["off_id_across"] = mean_relative(true[across, drain_column], predicted[across, drain_column])
        rows.append(row)
    table = pd.DataFrame(rows)
    # Each metric averaged over the cases it is defined for.
    table = pd.concat([table, pd.DataFrame([{"case": "all", **table[METRICS].mean().to_dict()}])], ignore_index=True)
    return table


def label_and_seed(run_dir: Path) -> tuple[str, int]:
    label, separator, seed = run_dir.name.rpartition("_s")
    if not separator or not seed.isdigit():
        raise ValueError(f"run directory {run_dir} is not named <label>_s<seed>")
    return label, int(seed)


def collect(runs_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(runs_root.glob("*/results/arm_metrics.csv")):
        label, seed = label_and_seed(path.parent.parent)
        overall = pd.read_csv(path).set_index("case").loc["all"]
        frames.append({"label": label, "seed": seed, **overall[METRICS].to_dict()})
    if not frames:
        raise FileNotFoundError(f"no */results/arm_metrics.csv under {runs_root}")
    return pd.DataFrame(frames)


def improved(arm: pd.Series, baseline_mean: pd.Series, baseline_std: pd.Series, metric: str, sigmas: float) -> bool:
    return bool(baseline_mean[metric] - arm[metric] > sigmas * baseline_std[metric])


def kill_verdict(arm_name: str, arm: pd.Series, baseline_mean: pd.Series, baseline_std: pd.Series,
                 sigmas: float) -> tuple[bool, str]:
    """Whether the arm's kill condition of the config fires, with the numbers behind it."""
    if arm_name == "A1":
        value = arm["on_ig_narrow_ratio"]
        return value >= 1.5, f"on_ig_narrow_ratio {value:.3f} (baseline {baseline_mean['on_ig_narrow_ratio']:.3f})"
    if arm_name == "A2":
        factor = baseline_mean["on_ig"] / arm["on_ig"]
        return factor < 1.5, f"on_ig improved {factor:.3f}x"
    if arm_name == "A3":
        hot = improved(arm, baseline_mean, baseline_std, "hot_score", sigmas)
        return (arm["off_id_across"] > 0.05 or not hot), \
            f"off_id_across {arm['off_id_across']:.4f}, hot_score improved {hot}"
    if arm_name == "A4":
        across = improved(arm, baseline_mean, baseline_std, "off_id_across", sigmas)
        return not across, f"off_id_across {arm['off_id_across']:.4f} improved {across}"
    if arm_name == "A5":
        ratio = arm["gatebias_score"] / baseline_mean["gatebias_score"]
        return ratio > 0.95, f"gatebias_score ratio to baseline {ratio:.3f}"
    raise ValueError(f"no kill condition for arm {arm_name}")


def report(runs_root: Path) -> None:
    config = tomllib.loads(CONFIG_PATH.read_text())
    sigmas = config["noise"]["improvement_sigmas"]
    table = collect(runs_root)
    baseline = table[table["label"] == "baseline"]
    if len(baseline) < 2:
        raise ValueError(f"need at least two baseline seeds for a spread, found {len(baseline)}")
    baseline_mean = baseline[METRICS].mean()
    baseline_std = baseline[METRICS].std(ddof=1)
    noise = pd.DataFrame({"mean": baseline_mean, "std": baseline_std, "min": baseline[METRICS].min(),
                          "max": baseline[METRICS].max(), "seeds": len(baseline)})
    noise.to_csv(runs_root / "noise_floor.csv", float_format="%.6g")
    print(f"baseline, {len(baseline)} seeds")
    print(noise.to_string(float_format=lambda value: f"{value:.5g}"))

    rows = []
    for label, group in table.groupby("label", sort=True):
        if label == "baseline":
            continue
        arm = group[METRICS].mean()
        row = {"label": label, "seeds": " ".join(str(seed) for seed in sorted(group["seed"])), "runs": len(group),
               **{f"{metric}_mean": arm[metric] for metric in METRICS},
               **{f"{metric}_std": group[metric].std(ddof=1) if len(group) > 1 else np.nan for metric in METRICS}}
        arm_name = label.split("_")[0]
        if arm_name in config["arms"]:
            fired, evidence = kill_verdict(arm_name, arm, baseline_mean, baseline_std, sigmas)
            row.update({"kill_condition": config["arms"][arm_name]["kill"], "killed": fired, "evidence": evidence})
        rows.append(row)
    arms = pd.DataFrame(rows)
    arms.to_csv(runs_root / "arm_report.csv", index=False, float_format="%.6g")
    if not arms.empty:
        print(arms[["label", "runs", "gatebias_score_mean", "gatebias_score_std", "on_ig_mean", "off_id_across_mean",
                    "hot_score_mean", "killed", "evidence"]].to_string(index=False))


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("metrics", "report"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "metrics":
        for run_dir in map(Path, sys.argv[2:]):
            table = run_metrics(run_dir)
            table.to_csv(run_dir / "results" / "arm_metrics.csv", index=False, float_format="%.6g")
            print(run_dir.name)
            print(table.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    else:
        report(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
