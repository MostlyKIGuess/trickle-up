"""Stage 4 blind evaluation of config/stage4_eval.toml: bootstrap over corners, paired difference, verdicts.

Usage: python3 scripts/12_stage4_eval.py <out_dir> <part_b_data_dir>

<out_dir> holds one folder per method (frozen, old, oracle), each with the gate_predictions.csv and
cla_predictions.csv that 07_gates_ml.py and 09_cla_ml.py wrote for it on the stage 4 corners.
<part_b_data_dir> is the data/ of the Part B corners, used only for the null control, which predicts
each gate (and each adder vector) as the median SPICE leakage over the Part B corners.

The error of a simulation is |predicted - SPICE| / SPICE, the Kirchhoff composition (leak_predicted).
The bootstrap resamples corners, not rows: each draw takes 400 corners with replacement and averages
the error over every simulation of the drawn corners. All methods share the same draws, so the paired
difference new minus old is computed draw by draw. Writes <out_dir>/stage4_summary.csv.
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT

CONFIG = tomllib.loads((ROOT / "config" / "stage4_eval.toml").read_text())
LEVELS = {"gate": ("gate_predictions.csv", ["gate", "inputs"]), "adder": ("cla_predictions.csv", ["vector"])}


def load(out_dir: Path, method: str, level: str) -> pd.DataFrame:
    file_name, _ = LEVELS[level]
    frame = pd.read_csv(out_dir / method / file_name, dtype={"inputs": str})
    if frame[["leak_spice", "leak_predicted"]].isna().any().any():
        raise ValueError(f"{method} {level} predictions contain NaN")
    return frame


def relative_error(frame: pd.DataFrame, column: str = "leak_predicted") -> np.ndarray:
    return np.abs(frame[column].to_numpy() - frame["leak_spice"].to_numpy()) / frame["leak_spice"].to_numpy()


def corner_sums(frame: pd.DataFrame, errors: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-corner error sum and simulation count, in the order of corners."""
    grouped = pd.DataFrame({"corner": frame["corner"].to_numpy(), "error": errors}).groupby("corner")["error"]
    sums = grouped.sum().reindex(corners)
    counts = grouped.count().reindex(corners)
    if sums.isna().any():
        raise ValueError("a corner has no simulations")
    return sums.to_numpy(), counts.to_numpy()


def log10_r2(frame: pd.DataFrame) -> float:
    spice = frame["leak_spice"].to_numpy()
    predicted = frame["leak_predicted"].to_numpy()
    if (spice <= 0).any() or (predicted <= 0).any():
        raise ValueError("R2 on log10 leakage needs positive SPICE and predicted leakage")
    true_log, predicted_log = np.log10(spice), np.log10(predicted)
    return float(1 - ((predicted_log - true_log) ** 2).sum() / ((true_log - true_log.mean()) ** 2).sum())


def null_prediction(frame: pd.DataFrame, part_b: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    medians = part_b.groupby(keys)["leak_spice"].median().rename("leak_predicted").reset_index()
    return frame[keys + ["corner", "leak_spice"]].merge(medians, on=keys, how="left", validate="many_to_one")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    out_dir, part_b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    evaluation = CONFIG["eval"]
    generator = np.random.default_rng(evaluation["bootstrap_seed"])
    lower_quantile = (1 - evaluation["confidence"]) / 2
    rows = []
    verdicts = {}
    for level, (_, keys) in LEVELS.items():
        frames = {method: load(out_dir, method, level) for method in ("frozen", "old", "oracle")}
        corners = np.sort(frames["frozen"]["corner"].unique())
        if len(corners) != evaluation["corners"]:
            raise ValueError(f"{level}: {len(corners)} corners, the config says {evaluation['corners']}")
        expected = evaluation["gate_simulations" if level == "gate" else "adder_simulations"]
        for method, frame in frames.items():
            if len(frame) != expected:
                raise ValueError(f"{level} {method}: {len(frame)} simulations, the config says {expected}")
        part_b_name = "gate_predictions_oracle.csv" if level == "gate" else "cla_predictions_oracle.csv"
        part_b = pd.read_csv(part_b_dir / part_b_name, dtype={"inputs": str})
        frames["null"] = null_prediction(frames["frozen"], part_b, keys)

        draws = generator.integers(0, len(corners), size=(evaluation["bootstrap"], len(corners)))
        bootstrap_means = {}
        for method, frame in frames.items():
            errors = relative_error(frame)
            sums, counts = corner_sums(frame, errors, corners)
            bootstrap_means[method] = 100 * sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            low, high = np.quantile(bootstrap_means[method], [lower_quantile, 1 - lower_quantile])
            rows.append({"level": level, "method": method, "simulations": len(frame),
                         "mean_error_percent": 100 * errors.mean(), "ci_low": low, "ci_high": high,
                         "median_error_percent": 100 * np.median(errors),
                         "p95_error_percent": 100 * np.quantile(errors, 0.95),
                         "max_error_percent": 100 * errors.max(),
                         "r2_log10": log10_r2(frame) if method != "null" else np.nan})
        difference = bootstrap_means["frozen"] - bootstrap_means["old"]
        low, high = np.quantile(difference, [lower_quantile, 1 - lower_quantile])
        point = rows[-4]["mean_error_percent"] - rows[-3]["mean_error_percent"]
        rows.append({"level": level, "method": "frozen minus old", "mean_error_percent": point,
                     "ci_low": low, "ci_high": high})
        by_method = {row["method"]: row for row in rows if row["level"] == level}
        verdicts[level] = {
            # The gate oracle feeds SPICE transistor currents through the composition, so it must match to
            # about reltol. The adder oracle sums SPICE gate leakages, the gate-sum composition control,
            # whose gap (0.030 % max on Part B) is reported, not held to 1e-5 %.
            "oracle_max_below_1e-5_percent": (by_method["oracle"]["max_error_percent"] < 1e-5
                                              if level == "gate" else np.nan),
            "meets_99": by_method["frozen"]["ci_high"] < 1.0,
            "improved": high < 0,
            # "Fail by a wide margin" of the config, fixed here before any stage 4 result: ten times the frozen mean.
            "null_fails_wide": by_method["null"]["mean_error_percent"] > 10 * by_method["frozen"]["mean_error_percent"],
        }
    verdicts["gate"]["guardrail_p95"] = (
        next(row for row in rows if row["level"] == "gate" and row["method"] == "frozen")["p95_error_percent"]
        <= next(row for row in rows if row["level"] == "gate" and row["method"] == "old")["p95_error_percent"])

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "stage4_summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(verdicts).to_csv(out_dir / "stage4_verdicts.csv")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print(pd.DataFrame(verdicts).to_string())


if __name__ == "__main__":
    main()
