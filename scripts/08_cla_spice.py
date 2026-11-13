"""SPICE reference for the 3-bit carry-lookahead adder: 128 input vectors at every corner.

Usage: python3 scripts/08_cla_spice.py [max_corners] [workers]

The adder is one flat transistor netlist built from the gate definitions in gates.py, with the model
cards at rgatemod = 0 like the gate decks (gates.without_gate_resistor). Before the
corners, a logic check simulates all 128 vectors at the nominal corner (VDD 1 V, 27 C, nominal
process) and stops unless s0 + 2 s1 + 4 s2 + 8 c3 = a + b + cin for every vector, with each output
read as 1 above VDD / 2.

Writes
  spice/cla3.cir                     the adder netlist, nominal corner, vector 0
  results/cla_logic_check.csv        the nominal logic check
  data/cla_sims.csv                  per vector and corner: leakage (device- and source-based), net voltages
  results/cla_device_vs_source.csv   device-based against source-based leakage
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import DATA_DIR, RESULTS_DIR, SPICE_DIR, SPICE_WORKERS
from gates import (CLA_INSTANCES, CLA_NETS, CLA_OUTPUTS, CLA_TRANSISTORS, VECTOR_COUNT, circuit_model_cards, cla_deck,
                   nominal_corner, vector_operands)
from spice import run_deck


def simulate_vector(job: tuple) -> dict:
    vector, corner = job
    deck, printed = cla_deck(vector, corner, circuit_model_cards(corner))
    values = run_deck(deck, printed)
    a_value, b_value, carry_in = vector_operands(vector)
    return {"corner": corner["corner"], "vector": vector, "a": a_value, "b": b_value, "cin": carry_in,
            "vdd": corner["vdd"], "temp": corner["temp"], **values}


def output_sum(simulations: pd.DataFrame) -> pd.Series:
    """s0 + 2 s1 + 4 s2 + 8 c3 with each output read as 1 above VDD / 2."""
    bits = [(simulations[f"v_{net}"] > simulations["vdd"] / 2).astype(int) for net in CLA_OUTPUTS]
    return bits[0] + 2 * bits[1] + 4 * bits[2] + 8 * bits[3]


def device_against_source(simulations: pd.DataFrame) -> pd.DataFrame:
    difference = simulations["leak_source"] - simulations["leak_device"]
    relative = np.abs(difference) / simulations["leak_device"]
    return pd.DataFrame([{
        "simulations": len(simulations),
        "median_leak_device_A": simulations["leak_device"].median(),
        "min_leak_device_A": simulations["leak_device"].min(),
        "median_relative_difference": relative.median(),
        "p95_relative_difference": relative.quantile(0.95),
        "max_relative_difference": relative.max(),
        "max_abs_difference_A": np.abs(difference).max(),
    }])


def main():
    max_corners = int(sys.argv[1]) if len(sys.argv) > 1 else None
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else SPICE_WORKERS
    corners = pd.read_csv(DATA_DIR / "corners.csv")
    if max_corners is not None:
        corners = corners.head(max_corners)
    print(f"CLA: {len(CLA_INSTANCES)} gates, {len(CLA_TRANSISTORS)} transistors, {len(CLA_NETS)} gate output nets")

    nominal_point = nominal_corner()
    SPICE_DIR.mkdir(exist_ok=True)
    deck, _ = cla_deck(0, nominal_point, circuit_model_cards(nominal_point))
    header = ("* Example netlist: 3-bit carry-lookahead adder at the nominal corner, vector 0 (every input at 0 V).\n"
              "* scripts/08_cla_spice.py writes one deck like this per vector and corner, with the corner's\n"
              "* process values in the model cards, its VDD and temperature, and each input at 0 or VDD.\n"
              "* The model cards are given/models/22nm_LP.pm with rgatemod = 0 (see README).\n")
    (SPICE_DIR / "cla3.cir").write_text(header + deck)

    RESULTS_DIR.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        nominal = pd.DataFrame(list(pool.map(simulate_vector, [(vector, nominal_point) for vector in range(VECTOR_COUNT)])))
        nominal["expected"] = nominal["a"] + nominal["b"] + nominal["cin"]
        nominal["spice_sum"] = output_sum(nominal)
        nominal["passed"] = nominal["expected"] == nominal["spice_sum"]
        check_columns = ["vector", "a", "b", "cin", "expected", "spice_sum", "passed",
                         *[f"v_{net}" for net in CLA_OUTPUTS], "leak_device", "leak_source"]
        nominal[check_columns].to_csv(RESULTS_DIR / "cla_logic_check.csv", index=False, float_format="%.6e")
        print(f"nominal logic check: {nominal['passed'].sum()} of {len(nominal)} vectors give a + b + cin")
        if not nominal["passed"].all():
            raise RuntimeError("the adder fails the logic check, see results/cla_logic_check.csv")

        jobs = [(vector, corner) for corner in corners.to_dict("records") for vector in range(VECTOR_COUNT)]
        simulations = pd.DataFrame(list(pool.map(simulate_vector, jobs, chunksize=4)))

    simulations.to_csv(DATA_DIR / "cla_sims.csv", index=False, float_format="%.12e")
    print(f"data/cla_sims.csv: {len(simulations)} rows")
    wrong_sum = (output_sum(simulations) != simulations["a"] + simulations["b"] + simulations["cin"]).sum()
    print(f"corner logic check: {wrong_sum} of {len(simulations)} simulations give a wrong sum")

    comparison = device_against_source(simulations)
    comparison.to_csv(RESULTS_DIR / "cla_device_vs_source.csv", index=False, float_format="%.6e")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
