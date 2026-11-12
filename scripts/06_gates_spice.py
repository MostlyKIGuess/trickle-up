"""SPICE reference for the five gates: 18 gate and input cases at every corner of data/corners.csv.

Usage: python3 scripts/06_gates_spice.py [max_corners] [workers]
       python3 scripts/06_gates_spice.py rgatemod|diffusion [corners] [workers]

The second form runs one check on the first corners of data/corners.csv and nothing else:
  rgatemod   the rgatemod comparison below, on that many corners
  diffusion  every gate case with and without AD, AS, PD and PS on the device lines, which the TA
             netlist leaves out (results/diffusion_comparison.csv)

Before the gates, a drain current check compares the device-based drain and source currents of
gates.device_current_lines with the drain and source voltage-source currents of single transistors,
in both orientations and all four cases. Drain and source have no internal resistors (rdsmod = 0), so
those source currents are exact. Writes results/drain_current_check.csv and stops if they disagree.

The gate decks use the model cards with rgatemod = 0 (gates.without_gate_resistor). A second check
runs every gate case at the first corners with rgatemod = 1 and 0 and records which converge and
how far the leakages differ.

Writes
  spice/<gate>.cir                    one example netlist per gate, nominal corner, inputs at 0
  results/rgatemod_comparison.csv     gate leakage with rgatemod = 1 and 0, first corners
  data/gate_sims.csv                  per simulation: leakage (device-based and source-based), node voltages
  data/gate_transistors.csv           per simulation and transistor: terminal voltages and currents
  results/gate_device_vs_source.csv   device-based against source-based leakage, per gate case
"""

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (CASES, DATA_DIR, L_MIN, RESULTS_DIR, SPICE_DIR, SPICE_WORKERS, TEMP_NOMINAL, VDD_NOMINAL,
                    SPICE_OPTIONS, diffusion_geometry)
from gates import (CURRENT_COLUMNS, GATE_CASES, GATES, GROUND, INTERNAL_NODES, SUPPLY, TERMINALS, Transistor,
                   circuit_model_cards, corner_process, device_current_lines, gate_deck, nominal_corner)
from spice import model_cards, ngspice_version, run_deck

# The rgatemod comparison runs on this many corners.
GATE_RESISTOR_CHECK_CORNERS = 200

# A drain or source current at or above this is checked against its source current.
CHECKED_CURRENT_FLOOR = 1e-15
CHECK_TOLERANCE = 1e-4


def drain_check_deck(case_name: str, temp: float, drain_fraction: float, source_fraction: float) -> str:
    case = CASES[case_name]
    width = 2 * L_MIN
    transistor = Transistor("m1", case["device"], "drain", "gate", "source", "body", width)
    sign = "" if case["device"] == "nmos" else "-"
    return "\n".join([
        f"* drain current check, {case_name}",
        model_cards({}),
        SPICE_OPTIONS,
        f".temp {temp:.6f}",
        f"m1 drain gate source body {case['device']} W={width:.6e} L={L_MIN:.6e} {diffusion_geometry(width)}",
        f"vd drain 0 {drain_fraction * VDD_NOMINAL:.9f}",
        f"vg gate 0 {case['gate'] * VDD_NOMINAL:.9f}",
        f"vs source 0 {source_fraction * VDD_NOMINAL:.9f}",
        f"vb body 0 {case['body'] * VDD_NOMINAL:.9f}",
        ".control",
        "set numdgt=12",
        "set num_threads=1",
        "op",
        *device_current_lines(transistor),
        "let id_source = -i(vd)",
        "let is_source = -i(vs)",
        # What the drain current would be if @m[id] were the drain terminal current.
        f"let id_naive = {sign}@m1[id]",
        "print m1_id m1_is id_source is_source id_naive",
        ".endc",
        ".end",
        "",
    ])


def run_drain_check(job: tuple) -> dict:
    case_name, temp, drain_fraction, source_fraction = job
    values = run_deck(drain_check_deck(*job), ["m1_id", "m1_is", "id_source", "is_source", "id_naive"])
    return {"case": case_name, "temp": temp, "vd": drain_fraction * VDD_NOMINAL, "vs": source_fraction * VDD_NOMINAL,
            **values}


def drain_current_check(pool: ProcessPoolExecutor) -> pd.DataFrame:
    fractions = [0.0, 0.5, 1.0]
    jobs = [(case_name, temp, drain_fraction, source_fraction)
            for case_name in CASES for temp in (-40.0, TEMP_NOMINAL, 125.0)
            for drain_fraction in fractions for source_fraction in fractions if drain_fraction != source_fraction]
    table = pd.DataFrame(list(pool.map(run_drain_check, jobs)))
    for terminal in ("id", "is"):
        table[f"{terminal}_relative_difference"] = (
            np.abs(table[f"m1_{terminal}"] - table[f"{terminal}_source"]) / np.abs(table[f"{terminal}_source"]))
    table["id_naive_relative_difference"] = (
        np.abs(table["id_naive"] - table["id_source"]) / np.abs(table["id_source"]))
    return table


def diffusion_leakage(job: tuple) -> dict:
    """Device-based leakage of one gate case with AD, AS, PD and PS on every device line or without them."""
    gate_name, inputs, corner, diffusion = job
    deck, printed = gate_deck(gate_name, inputs, corner, circuit_model_cards(corner), diffusion=diffusion)
    return {"gate": gate_name, "inputs": inputs, "corner": corner["corner"], "diffusion": int(diffusion),
            "leak_device": run_deck(deck, printed)["leak_device"]}


def diffusion_comparison(pool: ProcessPoolExecutor, corners: pd.DataFrame) -> pd.DataFrame:
    """Gate leakage with and without the diffusion geometry, against the stored data/gate_sims.csv.

    The TA netlist's device line has no AD, AS, PD or PS, so adding them was our own choice. The run
    with them must reproduce data/gate_sims.csv, which checks that this comparison builds the same decks.
    """
    jobs = [(gate_name, inputs, corner, diffusion) for gate_name, inputs in GATE_CASES
            for corner in corners.to_dict("records") for diffusion in (True, False)]
    runs = pd.DataFrame(list(pool.map(diffusion_leakage, jobs, chunksize=8)))
    table = runs.pivot(index=["gate", "inputs", "corner"], columns="diffusion", values="leak_device")
    table.columns = [f"leak_diffusion{column}" for column in table.columns]
    table = table.reset_index()
    stored = pd.read_csv(DATA_DIR / "gate_sims.csv", dtype={"inputs": str})[["gate", "inputs", "corner", "leak_device"]]
    table = table.merge(stored.rename(columns={"leak_device": "leak_stored"}), on=["gate", "inputs", "corner"],
                        how="left", validate="one_to_one")
    table["rerun_relative_difference"] = np.abs(table["leak_diffusion1"] - table["leak_stored"]) / table["leak_stored"]
    table["ratio_without_to_with"] = table["leak_diffusion0"] / table["leak_diffusion1"]
    return table


def simulate_gate(job: tuple) -> tuple[dict, list[dict]]:
    gate_name, inputs, corner = job
    gate = GATES[gate_name]
    deck, printed = gate_deck(gate_name, inputs, corner, circuit_model_cards(corner))
    values = run_deck(deck, printed)
    vdd = corner["vdd"]
    voltages = {GROUND: 0.0, SUPPLY: vdd}
    voltages |= {node: vdd * int(bit) for node, bit in zip(gate.inputs, inputs)}
    voltages |= {node: values[f"v_{node}"] for node in (gate.output, *gate.internal)}
    key = {"gate": gate_name, "inputs": inputs, "corner": corner["corner"]}
    simulation = {**key, "vdd": vdd, "temp": corner["temp"], "leak_device": values["leak_device"],
                  "leak_source": values["leak_source"], "v_out": voltages[gate.output]}
    simulation |= {f"v_{node}": voltages.get(node, np.nan) for node in INTERNAL_NODES}
    transistors = []
    for transistor in gate.transistors:
        row = {**key, "transistor": transistor.name, "device": transistor.device, "w": transistor.width}
        row |= {f"v{terminal[0]}": voltages[transistor.node(terminal)] for terminal in TERMINALS}
        row |= {current: values[f"{transistor.name}_{current}"] for current in CURRENT_COLUMNS}
        transistors.append(row)
    return simulation, transistors


def gate_resistor_leakage(job: tuple) -> dict:
    """Device-based leakage of one gate case with the model's gate resistor (rgatemod = 1) or without it."""
    gate_name, inputs, corner, gate_resistor = job
    model_text = model_cards(corner_process(corner)) if gate_resistor else circuit_model_cards(corner)
    deck, printed = gate_deck(gate_name, inputs, corner, model_text)
    leakage, source_leakage, status = np.nan, np.nan, "converged"
    try:
        values = run_deck(deck, printed)
        leakage, source_leakage = values["leak_device"], values["leak_source"]
    except RuntimeError as error:
        # With the gate resistor, a failed operating point is the result being recorded. Anything else,
        # and any failure without the gate resistor, stops the run.
        if not gate_resistor or "DC solution failed" not in str(error):
            raise
        status = "dc solution failed"
    except subprocess.TimeoutExpired:
        if not gate_resistor:
            raise
        status = "run_deck timeout"
    return {"gate": gate_name, "inputs": inputs, "corner": corner["corner"], "rgatemod": int(gate_resistor),
            "leak_device": leakage, "leak_source": source_leakage, "status": status}


def gate_resistor_comparison(pool: ProcessPoolExecutor, corners: pd.DataFrame) -> pd.DataFrame:
    """Leakage with rgatemod = 1 and 0 per gate case. A converged operating point whose device-based and
    source-based leakage disagree (source_difference) has not closed Kirchhoff's law at the supply."""
    jobs = [(gate_name, inputs, corner, gate_resistor) for gate_name, inputs in GATE_CASES
            for corner in corners.head(GATE_RESISTOR_CHECK_CORNERS).to_dict("records") for gate_resistor in (True, False)]
    runs = pd.DataFrame(list(pool.map(gate_resistor_leakage, jobs)))
    table = runs.pivot(index=["gate", "inputs", "corner"], columns="rgatemod",
                       values=["leak_device", "leak_source", "status"])
    table.columns = [f"{value.replace('leak_device', 'leak')}_rgatemod{column}" for value, column in table.columns]
    table = table.reset_index()
    for column in ("leak_rgatemod1", "leak_rgatemod0", "leak_source_rgatemod1", "leak_source_rgatemod0"):
        table[column] = table[column].astype(float)
    table["rgatemod1_converged"] = table["status_rgatemod1"] == "converged"
    table["rgatemod0_converged"] = table["status_rgatemod0"] == "converged"
    table["relative_difference"] = np.abs(table["leak_rgatemod0"] - table["leak_rgatemod1"]) / table["leak_rgatemod1"]
    for setting in ("rgatemod1", "rgatemod0"):
        table[f"source_difference_{setting}"] = (np.abs(table[f"leak_source_{setting}"] - table[f"leak_{setting}"])
                                                 / table[f"leak_{setting}"])
    return table


def write_example_netlists() -> None:
    SPICE_DIR.mkdir(exist_ok=True)
    corner = nominal_corner()
    for gate_name in GATES:
        inputs = "0" * len(GATES[gate_name].inputs)
        deck, _ = gate_deck(gate_name, inputs, corner, circuit_model_cards(corner))
        header = (f"* Example netlist: {gate_name} at the nominal corner with every input at 0 V.\n"
                  f"* scripts/06_gates_spice.py writes one deck like this per simulation, with the corner's\n"
                  f"* process values in the model cards, its VDD and temperature, and each input at 0 or VDD.\n"
                  f"* The model cards are given/models/22nm_LP.pm with rgatemod = 0 (see README).\n")
        (SPICE_DIR / f"{gate_name}.cir").write_text(header + deck)


def device_against_source(simulations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gate_name, inputs), group in simulations.groupby(["gate", "inputs"], sort=False):
        difference = group["leak_source"] - group["leak_device"]
        relative = np.abs(difference) / group["leak_device"]
        rows.append({"gate": gate_name, "inputs": inputs, "simulations": len(group),
                     "median_leak_device_A": group["leak_device"].median(),
                     "min_leak_device_A": group["leak_device"].min(),
                     "median_relative_difference": relative.median(),
                     "max_relative_difference": relative.max(),
                     "max_abs_difference_A": np.abs(difference).max()})
    return pd.DataFrame(rows)


def write_resistor_comparison(pool: ProcessPoolExecutor, corners: pd.DataFrame) -> None:
    resistor = gate_resistor_comparison(pool, corners)
    resistor.to_csv(RESULTS_DIR / "rgatemod_comparison.csv", index=False, float_format="%.12e")
    both = resistor["rgatemod1_converged"] & resistor["rgatemod0_converged"]
    print(f"rgatemod comparison: {len(resistor)} decks, rgatemod = 1 converged in "
          f"{resistor['rgatemod1_converged'].sum()}, rgatemod = 0 in {resistor['rgatemod0_converged'].sum()}, "
          f"largest relative difference where both converge {resistor.loc[both, 'relative_difference'].max():.3e}")


def write_diffusion_comparison(pool: ProcessPoolExecutor, corners: pd.DataFrame) -> None:
    table = diffusion_comparison(pool, corners)
    table.to_csv(RESULTS_DIR / "diffusion_comparison.csv", index=False, float_format="%.12e")
    rerun = table["rerun_relative_difference"].max()
    print(f"diffusion comparison: {len(table)} gate cases, rerun with the geometry against data/gate_sims.csv "
          f"largest relative difference {rerun:.3e}")
    if not rerun <= CHECK_TOLERANCE:
        raise RuntimeError(f"the rerun with the diffusion geometry differs from data/gate_sims.csv by {rerun:.3e}")
    summary = table.groupby(["gate", "inputs"], sort=False)["ratio_without_to_with"].describe(
        percentiles=[0.5])[["min", "50%", "max"]]
    print("leakage without / with the diffusion geometry, per gate case over the corners")
    print(summary.to_string())


def main():
    # Single checks on existing corners: python3 scripts/06_gates_spice.py rgatemod|diffusion [corners] [workers]
    if len(sys.argv) > 1 and sys.argv[1] in ("rgatemod", "diffusion"):
        check_name = sys.argv[1]
        corners = pd.read_csv(DATA_DIR / "corners.csv")
        if len(sys.argv) > 2:
            corners = corners.head(int(sys.argv[2]))
        workers = int(sys.argv[3]) if len(sys.argv) > 3 else SPICE_WORKERS
        with ProcessPoolExecutor(max_workers=workers) as pool:
            if check_name == "rgatemod":
                write_resistor_comparison(pool, corners)
            else:
                write_diffusion_comparison(pool, corners)
        return

    max_corners = int(sys.argv[1]) if len(sys.argv) > 1 else None
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else SPICE_WORKERS
    corners = pd.read_csv(DATA_DIR / "corners.csv")
    if max_corners is not None:
        corners = corners.head(max_corners)
    RESULTS_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "ngspice_version.txt").write_text(ngspice_version() + "\n")
    write_example_netlists()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        check = drain_current_check(pool)
        check.to_csv(RESULTS_DIR / "drain_current_check.csv", index=False, float_format="%.6e")
        print(check[["case", "temp", "vd", "vs", "id_source", "id_relative_difference", "is_relative_difference",
                     "id_naive_relative_difference"]].to_string(index=False))
        for terminal in ("id", "is"):
            checked = check[np.abs(check[f"{terminal}_source"]) >= CHECKED_CURRENT_FLOOR]
            worst = checked[f"{terminal}_relative_difference"].max()
            print(f"{terminal}: {len(checked)} of {len(check)} rows at or above {CHECKED_CURRENT_FLOOR:g} A, "
                  f"largest relative difference {worst:.3e}")
            if not worst <= CHECK_TOLERANCE:
                raise RuntimeError(f"device-based {terminal} differs from its source current by {worst:.3e}, "
                                   f"see results/drain_current_check.csv")

        write_resistor_comparison(pool, corners)

        jobs = [(gate_name, inputs, corner) for gate_name, inputs in GATE_CASES
                for corner in corners.to_dict("records")]
        results = list(pool.map(simulate_gate, jobs, chunksize=8))

    simulations = pd.DataFrame([simulation for simulation, _ in results])
    transistors = pd.DataFrame([row for _, rows in results for row in rows])
    simulations.to_csv(DATA_DIR / "gate_sims.csv", index=False, float_format="%.12e")
    transistors.to_csv(DATA_DIR / "gate_transistors.csv", index=False, float_format="%.12e")
    print(f"data/gate_sims.csv: {len(simulations)} rows, data/gate_transistors.csv: {len(transistors)} rows")

    comparison = device_against_source(simulations)
    comparison.to_csv(RESULTS_DIR / "gate_device_vs_source.csv", index=False, float_format="%.6e")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
