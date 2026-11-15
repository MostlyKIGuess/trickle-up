"""Error analysis for the report, from the data of the committed runs. No SPICE and no training.

Usage (from the repository root):
    nix develop -c python3 report/error_analysis.py [data_dir] [models_dir] [uniform_only_models_dir]

data_dir holds the gitignored data/ of the runs (gate_transistors.csv, gate_sims.csv, corners.csv,
gate_predictions_ml.csv, cla_predictions_ml.csv and the Part A datasets). models_dir holds the chosen
transistor models of results/chosen_models.csv. uniform_only_models_dir holds the first models,
trained on the uniform set alone (run train-1028-0145), and may be left out.

The saved models are only evaluated, on the transistor rows the Part B SPICE runs recorded. Writes to
report/analysis/:
  on_vds.csv                  |Vds| of the ON transistors of the Part B gates
  off_gate_current.csv        how small the OFF gate currents of each Part A set get
  transistor_errors.csv       per case, terminal and |Vds| class, the model error on the gate transistors
  on_gate_current_by_width.csv  gate-current error of the ON transistors per case and width
  attribution.csv             gate error with one case's model at a time and SPICE currents elsewhere,
                              and the share of the leakage that is ON-transistor gate current
  on_gate_share_by_temperature.csv  that share in five temperature bins
  corner_errors.csv           rank correlation of gate and adder error with temperature, VDD and vth0
  corner_bins.csv             the same, binned
  per_corner.csv              per corner: scaled corner variables, median gate and adder error and bias
  per_corner_correlations.csv rank correlations of the per-corner errors
  trend_sweeps.csv            decades of change in each one-variable SPICE sweep
  gate_records.csv            per gate simulation: SPICE and ML leakage, the output voltage and the
                              internal node voltages (the record Section 5 of the brief asks for)
  corner_error.{pdf,png}      the figure of the error analysis
  sample_vds.{pdf,png}        |Vds| of each Part A sample set against the transistors of the Part B gates
  current_ranges.{pdf,png}    range of every terminal current of each Part A sample set
  learning_curve.{pdf,png}    validation error against the number of training samples
  attribution.{pdf,png}       attribution.csv as a figure
"""

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common import (CASES, CORNER_COUNT, L_MIN, PROCESS_PARAMETERS, PROCESS_SPREAD, SAMPLE_SETS, TEMP_RANGE,  # noqa: E402
                    VDD_NOMINAL, VDD_SPREAD, nominal_process_values)
from fet_model import CURRENT_FLOOR, load_models  # noqa: E402
from gates import CURRENT_COLUMNS, GATE_COLORS  # noqa: E402

gates_ml = importlib.import_module("07_gates_ml")

DEFAULT_DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "report" / "analysis"
KEY = ["gate", "inputs", "corner"]
SMALL_VDS = 1e-3
BIN_COUNT = 5
DEVICE_COLORS = {"nmos": "#2a6fb0", "pmos": "#c0582b"}
# The set colours of the parity plots of 04_train.py.
SET_COLORS = {"uniform": "#2a6fb0", "targeted": "0.35", "gatebias": "#1baf7a"}
SET_LABELS = {"uniform": "uniform", "targeted": "targeted", "gatebias": "gate-bias"}
# --mask-off-ig of 04_train.py leaves OFF gate currents at or below this out of the loss.
MASKED_OFF_GATE_CURRENT = 1e-26
# |Vds| below this is drawn at this value in the sample figure.
SMALLEST_DRAWN_VDS = 1e-10
GATE_ORDER = ["inv", "nand2", "nor2", "and2", "or2"]
# Appended to the label of an error axis. In a y label, which is rotated, the left arrow points down:
# lower is better.
LOWER_IS_BETTER = " \u2190"


def save_figure(figure: plt.Figure, path_stem: Path) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(f"{path_stem}.{suffix}", dpi=200)
    plt.close(figure)


def sample_figure(data_dir: Path, transistors: pd.DataFrame, cases: pd.Series, path_stem: Path) -> None:
    """Cumulative share below each |Vds|: every sample set against the transistors of the Part B
    gate simulations. The dot is the share of the ON transistors in the gates below 1 mV."""

    def cumulative(axis: plt.Axes, vds: pd.Series, **style) -> None:
        ordered = np.sort(vds.clip(lower=SMALLEST_DRAWN_VDS).to_numpy())
        axis.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", **style)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for axis, state in zip(axes, ("on", "off")):
        in_gates = (transistors["vd"] - transistors["vs"]).abs()[cases.str.endswith(f"_{state}")]
        cumulative(axis, in_gates, color="black", linewidth=2.2, label="Part B gates")
        for set_name, suffix in SAMPLE_SETS.items():
            rows = pd.concat([pd.read_csv(data_dir / f"fet_{case_name}{suffix}.csv", usecols=["vd", "vs"])
                              for case_name in CASES if case_name.endswith(f"_{state}")])
            cumulative(axis, (rows["vd"] - rows["vs"]).abs(), color=SET_COLORS[set_name], linewidth=1.3,
                       label=SET_LABELS[set_name])
        if state == "on":
            axis.scatter([SMALL_VDS], [(in_gates < SMALL_VDS).mean()], s=28, color="black", zorder=4)
        axis.set_xscale("log")
        axis.set_xlim(SMALLEST_DRAWN_VDS, 1.2)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("|Vds| (V)")
        axis.set_title(f"{state.upper()} transistors", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Share below |Vds|")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(figure, path_stem)


def current_figure(data_dir: Path, path_stem: Path) -> None:
    """Range of every terminal current of every sample set: line min to max, bar 5th to 95th percentile,
    tick at the median. Currents below the floor of the log target are drawn at the floor."""
    figure, axes = plt.subplots(1, len(CASES), figsize=(7.2, 2.3), sharex=True, sharey=True)
    offsets = dict(zip(SAMPLE_SETS, (0.26, 0.0, -0.26)))
    for axis, case_name in zip(axes, CASES):
        for set_name, suffix in SAMPLE_SETS.items():
            currents = pd.read_csv(data_dir / f"fet_{case_name}{suffix}.csv", usecols=CURRENT_COLUMNS).abs()
            currents = currents.clip(lower=CURRENT_FLOOR)
            for row, column in enumerate(CURRENT_COLUMNS):
                height = len(CURRENT_COLUMNS) - 1 - row + offsets[set_name]
                low, median, high = np.percentile(currents[column], [5, 50, 95])
                axis.plot([currents[column].min(), currents[column].max()], [height, height],
                          color=SET_COLORS[set_name], linewidth=0.8, solid_capstyle="butt")
                axis.plot([low, high], [height, height], color=SET_COLORS[set_name], linewidth=4.5,
                          solid_capstyle="butt", label=SET_LABELS[set_name] if row == 0 else None)
                axis.plot([median, median], [height - 0.11, height + 0.11], color="white", linewidth=1.2)
        if case_name.endswith("_off"):
            gate_row = len(CURRENT_COLUMNS) - 1 - CURRENT_COLUMNS.index("ig")
            axis.plot([MASKED_OFF_GATE_CURRENT] * 2, [gate_row - 0.45, gate_row + 0.45], color="black",
                      linewidth=0.9, linestyle=(0, (3, 2)))
        axis.set_xscale("log")
        axis.set_xlim(CURRENT_FLOOR, 1e-2)
        axis.set_xticks([1e-30, 1e-20, 1e-10])
        axis.set_xlabel("|I| (A)")
        axis.set_title(case_name.replace("_", " ").upper(), fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(CURRENT_COLUMNS)), [column.capitalize() for column in reversed(CURRENT_COLUMNS)])
    axes[0].set_ylim(-0.6, len(CURRENT_COLUMNS) - 0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(figure, path_stem)


def attribution_figure(attribution: pd.DataFrame, path_stem: Path) -> None:
    """Median gate error per gate case: the bar with every model, a marker with one model at a time."""
    cases = attribution[attribution["gate"] != "all"].copy()
    cases["order"] = cases["gate"].map(GATE_ORDER.index)
    cases = cases.sort_values(["order", "inputs"]).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(7.2, 2.7))
    positions = np.arange(len(cases))
    axis.bar(positions, cases["all models"], width=0.72, color="#d0d0d0", label="all models", zorder=1)
    for case_name in CASES:
        device, state = case_name.split("_")
        axis.scatter(positions, cases[f"{case_name} only"], s=22, zorder=3, linewidths=1.1,
                     edgecolors=DEVICE_COLORS[device], facecolors=DEVICE_COLORS[device] if state == "on" else "white",
                     label=f"{device.upper()} {state.upper()} only")
    axis.set_xticks(positions, cases["inputs"])
    for gate_name, group in cases.groupby("gate", sort=False):
        axis.text(group.index.to_numpy().mean(), -0.2, gate_name.upper(), ha="center", va="top", fontsize=9,
                  transform=axis.get_xaxis_transform())
        if group.index[0] > 0:
            axis.axvline(group.index[0] - 0.5, color="#bbbbbb", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    axis.set_xlim(-0.6, len(cases) - 0.4)
    axis.set_ylabel("Median gate error (%)" + LOWER_IS_BETTER)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22), columnspacing=1.2,
                handletextpad=0.4)
    figure.tight_layout()
    save_figure(figure, path_stem)


def gate_records(simulations: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """SPICE and ML leakage of every gate simulation with the node voltages SPICE recorded.

    The ML prediction takes its node voltages from SPICE, as Step 4 of the brief prescribes, so the
    voltages of the two are the same numbers.
    """
    records = simulations.merge(predictions, on=KEY, validate="one_to_one")
    if len(records) != len(simulations) or not np.allclose(records["leak_device"], records["leak_spice"], rtol=1e-12):
        raise RuntimeError("the gate predictions do not belong to these gate simulations")
    records = records.rename(columns={"leak_device": "leak_spice_A", "leak_predicted": "leak_ml_A"})
    return records[[*KEY, "vdd", "temp", "leak_spice_A", "leak_ml_A", "v_out", "v_x", "v_y", "v_inner"]]


def on_vds_table(transistors: pd.DataFrame, cases: pd.Series) -> pd.DataFrame:
    vds = (transistors["vd"] - transistors["vs"]).abs()
    rows = []
    for state, selected in (("on", cases.str.endswith("_on")), ("off", cases.str.endswith("_off"))):
        values = vds[selected]
        rows.append({"state": state, "rows": len(values), "fraction_below_1mV": (values < SMALL_VDS).mean(),
                     "median_V": values.median(), "p90_V": values.quantile(0.9)})
    return pd.DataFrame(rows)


def off_gate_current_table(data_dir: Path) -> pd.DataFrame:
    rows = []
    for case_name in ("nmos_off", "pmos_off"):
        for set_name, suffix in (("uniform", ""), ("targeted", "_targeted")):
            gate_current = pd.read_csv(data_dir / f"fet_{case_name}{suffix}.csv")["ig"].abs()
            rows.append({"case": case_name, "set": set_name, "rows": len(gate_current),
                         "exactly_zero": int((gate_current == 0).sum()),
                         "below_floor_1e-30": int((gate_current < CURRENT_FLOOR).sum()),
                         "smallest_nonzero_A": gate_current[gate_current > 0].min(),
                         "median_A": gate_current.median()})
    return pd.DataFrame(rows)


def transistor_errors(transistors: pd.DataFrame, cases: pd.Series, predicted: pd.DataFrame, label: str) -> pd.DataFrame:
    small = (transistors["vd"] - transistors["vs"]).abs() < SMALL_VDS
    rows = []
    for (case_name, is_small), group in transistors.groupby([cases, small]):
        for column in CURRENT_COLUMNS:
            spice = group[column].to_numpy()
            model = predicted.loc[group.index, column].to_numpy()
            relative = np.abs(model - spice) / np.maximum(np.abs(spice), CURRENT_FLOOR)
            rows.append({"models": label, "case": case_name, "vds_below_1mV": bool(is_small), "rows": len(group),
                         "terminal": column, "median_rel_error": np.median(relative),
                         "p95_rel_error": np.percentile(relative, 95),
                         "median_abs_spice_A": np.median(np.abs(spice))})
    return pd.DataFrame(rows)


def on_gate_current_by_width(transistors: pd.DataFrame, cases: pd.Series, predicted: pd.DataFrame) -> pd.DataFrame:
    """Error of the predicted gate current of the ON transistors, per case and width in units of Lmin."""
    on = cases.str.endswith("_on")
    width = (transistors["w"] / L_MIN).round().astype(int)
    signed = predicted["ig"] / transistors["ig"] - 1
    rows = []
    for (case_name, width_in_lmin), group in transistors[on].groupby([cases[on], width[on]]):
        error = signed.loc[group.index]
        rows.append({"case": case_name, "width_in_lmin": width_in_lmin, "rows": len(group),
                     "median_abs_error_percent": 100 * error.abs().median(),
                     "median_bias_percent": 100 * error.median()})
    return pd.DataFrame(rows)


def gate_errors(transistors: pd.DataFrame, cases: pd.Series, currents: pd.DataFrame,
                simulations: pd.DataFrame) -> pd.Series:
    """Relative error of the Kirchhoff composition, indexed by gate, inputs and corner."""
    composed = gates_ml.compose_gate_leakage(transistors, gates_ml.balanced_currents(transistors, currents, cases))
    spice = simulations.set_index(KEY)["leak_device"]
    return (composed / spice.loc[composed.index] - 1).rename("error")


def attribution_table(transistors: pd.DataFrame, cases: pd.Series, predicted: pd.DataFrame,
                      simulations: pd.DataFrame) -> pd.DataFrame:
    """Median |error| per gate case with the model of one case only, SPICE currents for the rest."""
    spice_currents = transistors[CURRENT_COLUMNS].astype(float)
    variants = {"all models": predicted}
    for case_name in CASES:
        mixed = spice_currents.copy()
        selected = cases == case_name
        mixed.loc[selected] = predicted.loc[selected].to_numpy()
        variants[f"{case_name} only"] = mixed
    columns = {}
    for label, currents in variants.items():
        errors = gate_errors(transistors, cases, currents, simulations).abs()
        by_case = errors.groupby(level=["gate", "inputs"], sort=False).median()
        by_case.loc[("all", "all")] = errors.median()
        columns[label] = 100 * by_case
    # How much of the SPICE leakage is gate tunnelling of the ON transistors: the sum of their |Ig| over
    # the gate leakage, per simulation.
    on_gate = transistors["ig"].abs().where(cases.str.endswith("_on"), 0.0)
    on_gate = on_gate.groupby([transistors[column] for column in KEY]).sum()
    share = on_gate / simulations.set_index(KEY)["leak_device"].loc[on_gate.index]
    share_by_case = share.groupby(level=["gate", "inputs"], sort=False).median()
    share_by_case.loc[("all", "all")] = share.median()
    columns["on_gate_current_share_percent"] = 100 * share_by_case
    return pd.DataFrame(columns).reset_index(), share


def share_by_temperature(share: pd.Series, corners: pd.DataFrame) -> pd.DataFrame:
    """ON gate-current share of the leakage in five equal temperature bins, as in corner_bins.csv."""
    temperature = corners.set_index("corner")["temp"]
    values = temperature.loc[share.index.get_level_values("corner")].to_numpy()
    edges = np.linspace(temperature.min(), temperature.max(), BIN_COUNT + 1)
    bins = np.clip(np.digitize(values, edges[1:-1]), 0, BIN_COUNT - 1)
    frame = share.rename("share").reset_index()
    rows = []
    for index, group in frame.groupby(bins):
        lowest = group.loc[group["share"].idxmin()]
        rows.append({"temp_low": edges[index], "temp_high": edges[index + 1], "simulations": len(group),
                     "median_share_percent": 100 * group["share"].median(),
                     "min_share_percent": 100 * lowest["share"], "min_gate": lowest["gate"],
                     "min_inputs": lowest["inputs"], "min_corner": int(lowest["corner"]),
                     "min_temp": temperature.loc[lowest["corner"]]})
    return pd.DataFrame(rows)


def corner_tables(gate_predictions: pd.DataFrame, cla_predictions: pd.DataFrame,
                  corners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    nominal = nominal_process_values()
    corner_frame = corners.set_index("corner")
    variables = {
        "temp": corner_frame["temp"],
        "vdd": corner_frame["vdd"],
        "nmos_vth0_ratio": corner_frame["nmos_vth0"] / nominal["nmos"]["vth0"],
        "pmos_vth0_ratio": corner_frame["pmos_vth0"] / nominal["pmos"]["vth0"],
    }
    frames = {
        "gates": gate_predictions.assign(error=gate_predictions["leak_predicted"] / gate_predictions["leak_spice"] - 1),
        "adder": cla_predictions.assign(error=cla_predictions["leak_predicted"] / cla_predictions["leak_spice"] - 1),
    }
    correlation_rows, bin_rows = [], []
    for level, frame in frames.items():
        frame = frame.copy()
        frame["abs_error_percent"] = 100 * frame["error"].abs()
        for name, values in variables.items():
            frame[name] = values.loc[frame["corner"]].to_numpy()
            # Spearman: the ranks of |error| against the ranks of the corner variable.
            correlation_rows.append({"level": level, "variable": name,
                                     "spearman_abs_error": frame["abs_error_percent"].rank().corr(frame[name].rank()),
                                     "spearman_signed_error": frame["error"].rank().corr(frame[name].rank())})
            edges = np.linspace(values.min(), values.max(), BIN_COUNT + 1)
            bins = np.clip(np.digitize(frame[name], edges[1:-1]), 0, BIN_COUNT - 1)
            for index, group in frame.groupby(bins):
                bin_rows.append({"level": level, "variable": name, "bin_low": edges[index], "bin_high": edges[index + 1],
                                 "simulations": len(group),
                                 "median_abs_error_percent": group["abs_error_percent"].median(),
                                 "p95_abs_error_percent": group["abs_error_percent"].quantile(0.95),
                                 "median_bias_percent": 100 * group["error"].median(),
                                 "median_leak_spice_A": group["leak_spice"].median()})
        frames[level] = frame
    return pd.DataFrame(correlation_rows), pd.DataFrame(bin_rows), frames


def worst_corners(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Median corner conditions of the simulations with the largest |error|, against all of them."""
    columns = ["temp", "vdd", "nmos_vth0_ratio", "pmos_vth0_ratio", "abs_error_percent"]
    worst = frame.nlargest(count, "abs_error_percent")
    return pd.DataFrame({"all": frame[columns].median(), f"worst {count}": worst[columns].median()})


def trend_decades(sweeps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variable, case_name), group in sweeps.groupby(["variable", "case"], sort=False):
        terminal = "ig" if variable == "toxe" else "id"
        first, last = abs(group[terminal].iloc[0]), abs(group[terminal].iloc[-1])
        rows.append({"variable": variable, "case": case_name, "terminal": terminal,
                     "value_first": group["value"].iloc[0], "value_last": group["value"].iloc[-1],
                     "current_first_A": first, "current_last_A": last, "decades": np.log10(last / first)})
    return pd.DataFrame(rows)


def corner_distance(corners: pd.DataFrame) -> pd.DataFrame:
    """Each corner variable scaled to -1 at the low end of its sampled range and +1 at the high end.

    Process values are scaled by their own magnitude, so for vth0 +1 means |vth0| 10 % above nominal.
    Adds the root mean square over the 12 variables (rms_all) and over VDD and temperature (rms_vdd_temp).
    """
    nominal = nominal_process_values()
    scaled = pd.DataFrame(index=corners["corner"])
    corner_frame = corners.set_index("corner")
    scaled["vdd"] = (corner_frame["vdd"] / VDD_NOMINAL - 1) / VDD_SPREAD
    scaled["temp"] = (corner_frame["temp"] - sum(TEMP_RANGE) / 2) / ((TEMP_RANGE[1] - TEMP_RANGE[0]) / 2)
    for device in ("nmos", "pmos"):
        for parameter in PROCESS_PARAMETERS:
            scaled[f"{device}_{parameter}"] = (corner_frame[f"{device}_{parameter}"] / nominal[device][parameter] - 1) / PROCESS_SPREAD
    variables = list(scaled.columns)
    scaled["rms_all"] = np.sqrt((scaled[variables] ** 2).mean(axis=1))
    scaled["rms_vdd_temp"] = np.sqrt((scaled[["vdd", "temp"]] ** 2).mean(axis=1))
    return scaled


def per_corner_table(frames: dict, corners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median gate and adder error of each corner, its scaled corner variables, and rank correlations."""
    table = corner_distance(corners)
    for level, frame in frames.items():
        grouped = frame.groupby("corner")
        table[f"{level}_median_abs_error_percent"] = grouped["abs_error_percent"].median()
        table[f"{level}_median_bias_percent"] = 100 * grouped["error"].median()
    rows = [{"level": "adder against gates", "variable": "median_bias_percent",
             "spearman_corner_median_abs_error": table["adder_median_bias_percent"].rank().corr(
                 table["gates_median_bias_percent"].rank())}]
    for level in frames:
        error = table[f"{level}_median_abs_error_percent"]
        for variable in ("rms_all", "rms_vdd_temp", "vdd", "temp", "nmos_vth0", "pmos_vth0"):
            distance = table[variable].abs()
            rows.append({"level": level, "variable": variable,
                         "spearman_corner_median_abs_error": error.rank().corr(distance.rank())})
    return table.reset_index(), pd.DataFrame(rows)


def corner_figure(frames: dict, per_corner: pd.DataFrame, path_stem: Path) -> None:
    """Left: |error| of every gate simulation against temperature. Right: per-corner adder error."""
    gates = frames["gates"]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axis = axes[0]
    for gate_name, group in gates.groupby("gate", sort=False):
        axis.scatter(group["temp"], group["abs_error_percent"].clip(lower=1e-3), s=3, alpha=0.35,
                     color=GATE_COLORS[gate_name], label=gate_name, rasterized=True, linewidths=0)
    edges = np.linspace(TEMP_RANGE[0], TEMP_RANGE[1], 9)
    centers = (edges[:-1] + edges[1:]) / 2
    bins = np.clip(np.digitize(gates["temp"], edges[1:-1]), 0, len(centers) - 1)
    medians = gates.groupby(bins)["abs_error_percent"].median()
    high = gates.groupby(bins)["abs_error_percent"].quantile(0.95)
    axis.plot(centers[medians.index], medians.to_numpy(), color="black", linewidth=2, label="median")
    axis.plot(centers[high.index], high.to_numpy(), color="black", linewidth=1.2, linestyle="--", label="95th pct")
    axis.set_yscale("log")
    axis.set_ylim(1e-3, 50)
    axis.set_xlabel("Temperature (C)")
    axis.set_ylabel("Gate |ML - SPICE| / SPICE (%)" + LOWER_IS_BETTER)
    axis.legend(frameon=False, fontsize=6.5, markerscale=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4)

    axis = axes[1]
    limit = 1.1 * np.abs(per_corner[["gates_median_bias_percent", "adder_median_bias_percent"]]).to_numpy().max()
    axis.plot([-limit, limit], [-limit, limit], color="#888888", linewidth=1, zorder=1)
    axis.scatter(per_corner["gates_median_bias_percent"], per_corner["adder_median_bias_percent"], s=10,
                 color="#2a78d6", linewidths=0, zorder=2)
    worst = per_corner.nlargest(3, "adder_median_abs_error_percent").sort_values("gates_median_bias_percent")
    axis.scatter(worst["gates_median_bias_percent"], worst["adder_median_bias_percent"], s=42, facecolors="none",
                 edgecolors="black", linewidths=0.8, zorder=3)
    # Each label sits in the empty half under the diagonal with a leader line: straight below a corner
    # in the upper half, to the right of one in the lower half, where below would leave the axes.
    # Two named corners close together get different leader lengths.
    for index, (_, row) in enumerate(worst.iterrows()):
        point = (row["gates_median_bias_percent"], row["adder_median_bias_percent"])
        upper_half = point[1] > 0
        axis.annotate(f"corner {int(row['corner'])}", point,
                      xytext=(0, -26 - 14 * (index % 2)) if upper_half else (26, 0), textcoords="offset points",
                      ha="center" if upper_half else "left", va="top" if upper_half else "center", fontsize=7,
                      arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.6, "shrinkB": 4})
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.set_xlabel("Gate median error per corner (%)")
    axis.set_ylabel("Adder median error per corner (%)")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(f"{path_stem}.{suffix}", dpi=200)
    plt.close(figure)


def learning_figure(path_stem: Path) -> None:
    """Median error on the uniform validation samples against the number of training samples, one line
    per case, as a mean over the four terminals. From the learning curve of 04_train.py, which refits
    the model that the comparison selected for each case."""
    curve = pd.read_csv(ROOT / "results" / "comparison" / "learning_curve.csv")
    curve = curve[curve["set"] == "uniform"]
    errors = 100 * curve.groupby(["case", "train_size"])["median_rel_error"].mean().unstack("case")
    largest = errors.index.max()
    # The two ON lines end close together, so their labels are moved apart.
    label_shift = {"nmos_on": 1.12, "pmos_on": 0.9}
    figure, axis = plt.subplots(figsize=(4.6, 2.5))
    for case_name in CASES:
        device, state = case_name.split("_")
        axis.plot(errors.index, errors[case_name], color=DEVICE_COLORS[device], linewidth=1.6,
                  linestyle="-" if state == "on" else (0, (4, 2)), marker="o", markersize=4.5,
                  markerfacecolor=DEVICE_COLORS[device] if state == "on" else "white")
        axis.text(largest * 1.04, errors.loc[largest, case_name] * label_shift.get(case_name, 1.0),
                  case_name.replace("_", " ").upper(), color="0.15", fontsize=8, va="center")
    axis.set_yscale("log")
    axis.set_ylim(bottom=0.4)
    axis.set_xlim(900, largest * 1.3)
    axis.set_xticks(list(errors.index), [f"{size:,}" for size in errors.index])
    axis.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    axis.minorticks_off()
    axis.set_xlabel("Training samples per case")
    axis.set_ylabel("Median error (%)" + LOWER_IS_BETTER)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, path_stem)


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    models_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models"
    uniform_only_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transistors = pd.read_csv(data_dir / "gate_transistors.csv", dtype={"inputs": str})
    simulations = pd.read_csv(data_dir / "gate_sims.csv", dtype={"inputs": str})
    corners = pd.read_csv(data_dir / "corners.csv")
    if len(corners) != CORNER_COUNT:
        raise ValueError(f"expected {CORNER_COUNT} corners, found {len(corners)}")
    inputs = gates_ml.model_inputs(transistors, corners)
    cases = gates_ml.transistor_cases(transistors, inputs)

    on_vds = on_vds_table(transistors, cases)
    on_vds.to_csv(OUTPUT_DIR / "on_vds.csv", index=False, float_format="%.6g")
    print(on_vds.to_string(index=False))
    sample_figure(data_dir, transistors, cases, OUTPUT_DIR / "sample_vds")
    current_figure(data_dir, OUTPUT_DIR / "current_ranges")
    learning_figure(OUTPUT_DIR / "learning_curve")

    off_gate = off_gate_current_table(data_dir)
    off_gate.to_csv(OUTPUT_DIR / "off_gate_current.csv", index=False, float_format="%.6g")
    print(off_gate.to_string(index=False))

    predicted = gates_ml.predict_currents(load_models(models_dir), inputs, cases)
    errors = [transistor_errors(transistors, cases, predicted, "chosen")]
    if uniform_only_dir is not None:
        uniform_only = gates_ml.predict_currents(load_models(uniform_only_dir), inputs, cases)
        errors.append(transistor_errors(transistors, cases, uniform_only, "uniform only"))
    by_width = on_gate_current_by_width(transistors, cases, predicted)
    by_width.to_csv(OUTPUT_DIR / "on_gate_current_by_width.csv", index=False, float_format="%.6g")
    print(by_width.to_string(index=False))
    errors = pd.concat(errors, ignore_index=True)
    errors.to_csv(OUTPUT_DIR / "transistor_errors.csv", index=False, float_format="%.6g")
    print(errors.to_string(index=False))

    attribution, share = attribution_table(transistors, cases, predicted, simulations)
    attribution.to_csv(OUTPUT_DIR / "attribution.csv", index=False, float_format="%.6g")
    print(attribution.to_string(index=False))
    attribution_figure(attribution, OUTPUT_DIR / "attribution")
    share_table = share_by_temperature(share, corners)
    share_table.to_csv(OUTPUT_DIR / "on_gate_share_by_temperature.csv", index=False, float_format="%.6g")
    print(share_table.to_string(index=False))
    # The all-models column must reproduce results/gate_results_ml.csv.
    committed = pd.read_csv(ROOT / "results" / "gate_results_ml.csv", dtype={"inputs": str})
    matched = committed.merge(attribution, on=["gate", "inputs"], validate="one_to_one")
    if len(matched) != len(committed):
        raise RuntimeError("gate cases of the attribution and results/gate_results_ml.csv differ")
    difference = np.abs(matched["all models"] - matched["median_error_percent"]).max()
    print(f"largest difference from results/gate_results_ml.csv: {difference:.3g} points")
    if difference > 1e-3:
        raise RuntimeError("the recomposed gate errors do not match results/gate_results_ml.csv")

    gate_predictions = pd.read_csv(data_dir / "gate_predictions_ml.csv", dtype={"inputs": str})
    records = gate_records(simulations, gate_predictions)
    # The records must be the predictions behind results/gate_results_ml.csv.
    mean_error = 100 * (records["leak_ml_A"] / records["leak_spice_A"] - 1).abs().mean()
    committed_mean = committed.loc[committed["gate"] == "all", "mean_error_percent"].iloc[0]
    if abs(mean_error - committed_mean) > 1e-4:
        raise RuntimeError(f"gate records give {mean_error} % mean error, results/gate_results_ml.csv {committed_mean} %")
    records.to_csv(OUTPUT_DIR / "gate_records.csv", index=False, float_format="%.12g")
    cla_predictions = pd.read_csv(data_dir / "cla_predictions_ml.csv")
    correlations, bins, frames = corner_tables(gate_predictions, cla_predictions, corners)
    correlations.to_csv(OUTPUT_DIR / "corner_errors.csv", index=False, float_format="%.4g")
    bins.to_csv(OUTPUT_DIR / "corner_bins.csv", index=False, float_format="%.6g")
    print(correlations.to_string(index=False))
    print(bins.to_string(index=False))
    for level, count in (("gates", 36), ("adder", 256)):
        print(level)
        print(worst_corners(frames[level], count).to_string())
        worst = frames[level].nlargest(count, "abs_error_percent")
        print(worst.groupby(["gate", "inputs"] if level == "gates" else "corner").size()
              .sort_values(ascending=False).head(6).to_string())
    per_corner, corner_correlations = per_corner_table(frames, corners)
    per_corner.to_csv(OUTPUT_DIR / "per_corner.csv", index=False, float_format="%.6g")
    corner_correlations.to_csv(OUTPUT_DIR / "per_corner_correlations.csv", index=False, float_format="%.4g")
    print(corner_correlations.to_string(index=False))
    print(per_corner.nlargest(3, "adder_median_abs_error_percent").round(3).to_string(index=False))
    print(per_corner[["adder_median_abs_error_percent", "gates_median_abs_error_percent"]].describe().to_string())
    corner_figure(frames, per_corner, OUTPUT_DIR / "corner_error")

    sweeps = trend_decades(pd.read_csv(ROOT / "results" / "trend_sweeps.csv"))
    sweeps.to_csv(OUTPUT_DIR / "trend_sweeps.csv", index=False, float_format="%.6g")
    print(sweeps.to_string(index=False))


if __name__ == "__main__":
    main()
