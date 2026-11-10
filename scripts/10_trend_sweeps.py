"""One-variable-at-a-time SPICE sweeps behind the trends of the brief, for the report.

Usage: python3 scripts/10_trend_sweeps.py

The scatter plots of 03_check_data.py draw every input at random, so the vth0 trend there is hidden
under the spread of Vd, Vs and W. Here one transistor of width 4 Lmin sits at a fixed bias and only
one quantity moves, the rest held at nominal (VDD = 1 V, 27 C, the model file's process values):
- OFF channel leakage against temperature, -40 to 125 C
- OFF channel leakage against vth0, nominal +-10 %
- ON gate current against toxe, nominal +-10 %
The OFF devices have the full supply across them (NMOS drain at VDD, source at 0, PMOS the mirror).
The ON devices have source and drain at the same rail, so only tunnelling flows through the gate.

Writes results/trend_sweeps.csv and results/figures/trend_sweep.{pdf,png}.

Usage: python3 scripts/10_trend_sweeps.py width

Sweeps W alone from 1 to 8 Lmin in all four cases at the same bias, everything else nominal, and
writes results/width_sweep.csv. For each case and terminal it prints the spread (largest over smallest)
of I / W and of I / (W - 2 wint) over the sweep. A current that scales with the effective width has a
spread of 1 in the second. This checks the effective-width target of 04_train.py --weff on single
transistors, without the gate simulations.
"""

import sys

import importlib
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CASES, FIGURE_DIR, L_MIN, PROCESS_PARAMETERS, RESULTS_DIR, SPICE_WORKERS, TEMP_NOMINAL,
                    TEMP_RANGE, VDD_NOMINAL, WIDTH_RANGE, channel_width_offsets, nominal_process_values)
from spice import run_deck

sim_fet = importlib.import_module("02_sim_fet")

WIDTH = 4 * L_MIN
POINTS = 34
SPREAD = np.linspace(0.9, 1.1, 21)
COLORS = {"nmos": "#2a6fb0", "pmos": "#c0582b"}


def bias(case_name):
    """Terminal voltages of the sweep: full supply across an OFF device, none across an ON one."""
    case = CASES[case_name]
    high_side = VDD_NOMINAL if case["device"] == "nmos" else 0.0
    low_side = VDD_NOMINAL - high_side
    if case_name.endswith("_off"):
        vd, vs = high_side, low_side
    else:
        vd = vs = low_side
    return {"vdd": VDD_NOMINAL, "vd": vd, "vs": vs, "vg": case["gate"] * VDD_NOMINAL,
            "vb": case["body"] * VDD_NOMINAL}


def sweep_jobs():
    nominal = nominal_process_values()
    jobs = []
    sweeps = [("temp", case) for case in ("nmos_off", "pmos_off")] + \
             [("vth0", case) for case in ("nmos_off", "pmos_off")] + \
             [("toxe", case) for case in ("nmos_on", "pmos_on")]
    for variable, case_name in sweeps:
        device = CASES[case_name]["device"]
        if variable == "temp":
            values = np.linspace(*TEMP_RANGE, POINTS)
        else:
            values = nominal[device][variable] * SPREAD
        for value in values:
            sample = {"temp": TEMP_NOMINAL, "w": WIDTH, **bias(case_name), **nominal[device]}
            sample[variable] = value
            jobs.append({"variable": variable, "case": case_name, "device": device, "value": value, **sample})
    return jobs


def simulate(job):
    return run_deck(sim_fet.fet_deck(job, job["device"]), sim_fet.CURRENT_NAMES)


def width_sweep():
    nominal = nominal_process_values()
    offsets = channel_width_offsets()
    jobs = []
    for case_name, case in CASES.items():
        for width in np.linspace(*WIDTH_RANGE, 15):
            jobs.append({"case": case_name, "device": case["device"], "temp": TEMP_NOMINAL, "w": width,
                         **bias(case_name), **nominal[case["device"]]})
    with ProcessPoolExecutor(max_workers=SPICE_WORKERS) as pool:
        currents = list(pool.map(simulate, jobs))
    table = pd.DataFrame(jobs)[["case", "device", "w"]]
    table[["id", "ig", "is", "ib"]] = pd.DataFrame(currents).to_numpy()
    table["weff"] = table["w"] - 2 * table["device"].map(offsets)
    table.to_csv(RESULTS_DIR / "width_sweep.csv", index=False, float_format="%.10e")
    for case_name, rows in table.groupby("case", sort=False):
        for terminal in ("id", "ig", "is", "ib"):
            magnitude = np.abs(rows[terminal])
            if (magnitude == 0).any():
                print(f"{case_name:9s} {terminal}: zero current in the sweep, no spread")
                continue
            per_width = magnitude / rows["w"]
            per_effective_width = magnitude / rows["weff"]
            print(f"{case_name:9s} {terminal}: spread of I/W {per_width.max() / per_width.min():.4f}, "
                  f"of I/(W - 2 wint) {per_effective_width.max() / per_effective_width.min():.4f}")


def main():
    if sys.argv[1:] == ["width"]:
        width_sweep()
        return
    if sys.argv[1:]:
        raise SystemExit(__doc__)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = sweep_jobs()
    with ProcessPoolExecutor(max_workers=SPICE_WORKERS) as pool:
        currents = list(pool.map(simulate, jobs))
    table = pd.DataFrame(jobs)[["variable", "case", "device", "value", "temp", *PROCESS_PARAMETERS]]
    table[["id", "ig", "is", "ib"]] = pd.DataFrame(currents).to_numpy()
    table.to_csv(RESULTS_DIR / "trend_sweeps.csv", index=False, float_format="%.10e")

    panels = [("temp", "id", "Temperature (C)", "OFF current |Id| (A)"),
              ("vth0", "id", "|vth0| / nominal", "OFF current |Id| (A)"),
              ("toxe", "ig", "toxe / nominal", "ON gate current |Ig| (A)")]
    figure, axes = plt.subplots(1, 3, figsize=(10, 3))
    for axis, (variable, terminal, x_label, y_label) in zip(axes, panels):
        for case_name, rows in table[table["variable"] == variable].groupby("case", sort=False):
            device = CASES[case_name]["device"]
            x_values = rows["value"] if variable == "temp" else rows["value"] / rows["value"].iloc[len(rows) // 2]
            axis.plot(x_values, np.abs(rows[terminal]), marker="o", markersize=3, color=COLORS[device],
                      label=device.upper())
        axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURE_DIR / f"trend_sweep.{suffix}", dpi=200)
    plt.close(figure)

    for (variable, case_name), rows in table.groupby(["variable", "case"], sort=False):
        terminal = "ig" if variable == "toxe" else "id"
        first, last = np.abs(rows[terminal].iloc[0]), np.abs(rows[terminal].iloc[-1])
        print(f"{variable:5s} {case_name:9s} |{terminal}| {first:.3e} A to {last:.3e} A "
              f"({np.log10(last / first):+.2f} decades)")


if __name__ == "__main__":
    main()
