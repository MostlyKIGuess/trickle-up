"""Condense the model comparison and the frozen models' test errors into the report's tables.

Usage (from the repository root): nix develop -c python3 report/summarize_models.py

Reads results/comparison/ (the four model types and two targets of the brief, 04_train.py --output
data/comparison) and results/chosen_models.csv (the frozen method of config/stage4_frozen.toml).

Writes to report/analysis/:
  model_summary.csv   per case, model type and transform (and the null arm), per set: R2 on log10|I| and
                      on asinh(I / I0), median and 95th-percentile relative error, each the mean over the
                      four terminals, plus the
                      median selection score of 04_train.py (mean median error over terminals and the
                      uniform and targeted sets) and the gate-bias score of the validation rows
  chosen_summary.csv  per case and set, the frozen models' test median and 95th-percentile error of each
                      terminal
  arm_scores.{pdf,png}  the gate-bias score of every stage 2 and 3 run (results/arms/runs/), against the
                      band the noise rule of config/stage3_arms.toml draws around the baseline
"""

import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"
OUTPUT_DIR = ROOT / "report" / "analysis"
METRICS = ["r2_log10", "r2_asinh", "median_rel_error", "p95_rel_error"]
# The sets 04_train.py's median score averages over, its default training sets.
SCORED_SETS = ["uniform", "targeted"]
ARM_ORDER = ["baseline", "A1", "A2", "A3", "A4", "A5", "A2A4", "A2A4A5"]
KILLED_ARMS = {"A1", "A3"}
FROZEN_ARM = "A2A4A5"


def arm_figure(path_stem: Path) -> None:
    """Gate-bias score of every run, one column per arm, with the baseline's noise band."""
    config = tomllib.loads((ROOT / "config" / "stage3_arms.toml").read_text())
    sigmas = config["noise"]["improvement_sigmas"]
    scores = []
    for path in sorted((RESULTS_DIR / "arms" / "runs").glob("*.csv")):
        label = path.stem.rsplit("_s", 1)[0]
        row = pd.read_csv(path).set_index("case").loc["all"]
        scores.append({"label": label, "score": row["gatebias_score"]})
    scores = pd.DataFrame(scores)
    baseline = scores.loc[scores["label"] == "baseline", "score"]
    mean, deviation = baseline.mean(), baseline.std()
    figure, axis = plt.subplots(figsize=(7.0, 2.6))
    # The single arms on the left, their combinations on the right.
    axis.axvline(ARM_ORDER.index("A2A4") - 0.5, color="#bbbbbb", linewidth=0.8, linestyle=(0, (3, 3)))
    axis.axhspan(mean - sigmas * deviation, mean + sigmas * deviation, color="#d9d9d9", linewidth=0, zorder=0)
    axis.axhline(mean, color="#888888", linewidth=0.8, zorder=1)
    for position, label in enumerate(ARM_ORDER):
        values = scores.loc[scores["label"] == label, "score"].to_numpy()
        if len(values) == 0:
            raise ValueError(f"no runs of {label} in results/arms/runs")
        color = "#9a9a9a" if label in KILLED_ARMS else ("#e2231a" if label == FROZEN_ARM else "#2a6fb0")
        axis.scatter([position] * len(values), values, s=16, color=color, zorder=3, linewidths=0)
        axis.plot([position - 0.25, position + 0.25], [values.mean()] * 2, color=color, linewidth=1.6, zorder=2)
    axis.set_xticks(range(len(ARM_ORDER)), ["Baseline", "A1", "A2", "A3", "A4", "A5", "A2+A4", "A2+A4+A5"])
    axis.set_xlim(-0.6, len(ARM_ORDER) - 0.4)
    axis.set_ylim(0, None)
    # The label is rotated, so the left arrow points down: lower is better.
    axis.set_ylabel("Gate-bias score \u2190")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(f"{path_stem}.{suffix}", dpi=200)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(COMPARISON_DIR / "model_comparison.csv")
    if comparison.groupby(["case", "set", "model", "transform"]).size().ne(4).any():
        raise ValueError("expected four terminals per case, set, model and transform")
    means = comparison.groupby(["case", "model", "transform", "set"], sort=False)[METRICS].mean().unstack("set")
    means.columns = [f"{set_name}_{metric}" for metric, set_name in means.columns]
    scored = comparison[comparison["set"].isin(SCORED_SETS)]
    means["selection_score"] = scored.groupby(["case", "model", "transform"], sort=False)["median_rel_error"].mean()
    # 04_train.py records one fit time per model, repeated on each of its rows.
    means["fit_seconds"] = comparison.groupby(["case", "model", "transform"], sort=False)["fit_seconds"].first()
    means = means.reset_index()
    means["chosen"] = False
    chosen = pd.read_csv(COMPARISON_DIR / "chosen_models.csv")
    for case_name, rows in chosen.groupby("case", sort=False):
        model, transform = rows["model"].iloc[0], rows["transform"].iloc[0]
        selected = (means["case"] == case_name) & (means["model"] == model) & (means["transform"] == transform)
        if selected.sum() != 1:
            raise ValueError(f"{case_name}: chosen model {model} {transform} not in the comparison")
        score = means.loc[selected, "selection_score"].iloc[0]
        recorded = rows["validation_median_score"].iloc[0]
        if abs(score - recorded) > 1e-5:
            raise ValueError(f"{case_name}: selection score {score} differs from {recorded}")
        others = means[(means["case"] == case_name) & (means["model"] != "null_median")]
        if score != others["selection_score"].min():
            raise ValueError(f"{case_name}: the chosen model does not have the lowest selection score")
        means.loc[selected, "chosen"] = True
    means.to_csv(OUTPUT_DIR / "model_summary.csv", index=False, float_format="%.6g")
    print(means.to_string(index=False))

    frozen = pd.read_csv(RESULTS_DIR / "chosen_models.csv")
    test = frozen.pivot_table(index=["case", "set"], columns="terminal", values=["median_rel_error", "p95_rel_error"],
                              sort=False)
    test.columns = [f"{terminal}_{metric}" for metric, terminal in test.columns]
    test = test.reset_index()
    test.to_csv(OUTPUT_DIR / "chosen_summary.csv", index=False, float_format="%.6g")
    print(test.to_string(index=False))
    arm_figure(OUTPUT_DIR / "arm_scores")


if __name__ == "__main__":
    main()
