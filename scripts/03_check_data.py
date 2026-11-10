"""Kirchhoff check of both sample sets, and the three trend plots of the brief from the uniform set.

Usage: python3 scripts/03_check_data.py

Writes results/kirchhoff.csv and results/figures/trend_*.{pdf,png}.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CASES, DATA_DIR, ROOT, SAMPLE_SETS

RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TERMINALS = ["id", "ig", "is", "ib"]
COLORS = {"nmos": "#2a6fb0", "pmos": "#c0582b"}


def kirchhoff_table(datasets):
    rows = []
    for case_name, data in datasets.items():
        currents = data[TERMINALS].to_numpy()
        residual = np.abs(currents.sum(axis=1))
        # Relative to the largest terminal current of the same sample, so ON and OFF cases compare.
        relative = residual / np.abs(currents).max(axis=1)
        rows.append({
            "case": case_name,
            "samples": len(data),
            "median_abs_residual_A": np.median(residual),
            "max_abs_residual_A": residual.max(),
            "median_relative_residual": np.median(relative),
            "max_relative_residual": relative.max(),
        })
    return pd.DataFrame(rows)


def off_current(data):
    """Channel leakage of an OFF device: the drain current magnitude, only where |Vds| is large.

    With Vd and Vs drawn independently, many samples have Vds near zero and almost no channel
    current, so the trend plots keep the samples with |Vds| > VDD / 2.
    """
    keep = np.abs(data["vd"] - data["vs"]) > data["vdd"] / 2
    return data[keep], np.abs(data.loc[keep, "id"])


def scatter_panel(axis, datasets, x_column, y_of, x_label, y_label, case_names):
    for case_name in case_names:
        data, current = y_of(datasets[case_name])
        device = CASES[case_name]["device"]
        x_values = np.abs(data[x_column]) if x_column == "vth0" else data[x_column]
        axis.scatter(x_values, current, s=3, alpha=0.4, color=COLORS[device], label=device.upper(), rasterized=True)
    axis.set_yscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend(frameon=False, markerscale=4)


def save(figure, name):
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURE_DIR / f"{name}.{suffix}", dpi=200)
    plt.close(figure)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {case_name: pd.read_csv(DATA_DIR / f"fet_{case_name}.csv") for case_name in CASES}

    tables = []
    for set_name, suffix in SAMPLE_SETS.items():
        set_data = {case_name: pd.read_csv(DATA_DIR / f"fet_{case_name}{suffix}.csv") for case_name in CASES}
        tables.append(kirchhoff_table(set_data).assign(set=set_name))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(RESULTS_DIR / "kirchhoff.csv", index=False)
    print(table.to_string(index=False))

    off_cases = ["nmos_off", "pmos_off"]
    figure, axis = plt.subplots(figsize=(4, 3))
    scatter_panel(axis, datasets, "temp", off_current, "Temperature (C)", "OFF current |Id| (A)", off_cases)
    save(figure, "trend_off_vs_temp")

    figure, axis = plt.subplots(figsize=(4, 3))
    scatter_panel(axis, datasets, "vth0", off_current, "|vth0| (V)", "OFF current |Id| (A)", off_cases)
    save(figure, "trend_off_vs_vth0")

    # Gate tunnelling is largest with the channel inverted, so the ON cases show it best.
    figure, axis = plt.subplots(figsize=(4, 3))
    scatter_panel(axis, datasets, "toxe", lambda data: (data, np.abs(data["ig"])), "toxe (m)", "Gate current |Ig| (A)",
                  ["nmos_on", "pmos_on"])
    save(figure, "trend_ig_vs_toxe")


if __name__ == "__main__":
    main()
