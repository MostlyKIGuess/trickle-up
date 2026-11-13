"""Adder leakage composed from the gate results of Part B, against the full SPICE adder.

Usage: python3 scripts/09_cla_ml.py oracle
       python3 scripts/09_cla_ml.py ml

For each input vector, the logic value at every gate input picks that gate's Part B simulation at the
same corner and input combination, and the adder leakage is the sum of those gate leakages. The gate
leakages come from data/gate_predictions_<name>.csv, written by 07_gates_ml.py.

Two sums are compared with the device-based SPICE leakage of the full adder:
  composition control  the Part B SPICE gate leakages. Its error is the cost of composing gates at
                       all, from loading between gates and gate inputs that sit off the rails.
  <name>               the composed gate leakages of the predictor (oracle or ml), Kirchhoff
                       composition (cla_results_<name>.csv) and the per-terminal baseline
                       (cla_results_<name>_terminal.csv), see 07_gates_ml.py.

Before that, every gate output net of the SPICE adder at every corner is checked against the Python
logic evaluation that picks the gate input combinations.

Writes
  results/cla_net_logic_check.csv
  results/cla_composition_control.csv
  results/cla_results_<name>[_terminal].csv, results/cla_results_by_vector_<name>.csv
  data/cla_predictions_<name>.csv
  results/figures/cla_parity_<name>.{pdf,png}
"""

import sys

import numpy as np
import pandas as pd

from common import DATA_DIR, FIGURE_DIR, RESULTS_DIR
from gates import CLA_NETS, GATE_CASES, VECTOR_COUNT, cla_gate_cases, cla_logic, comparison_metrics, parity_plot


def net_logic_check(simulations: pd.DataFrame) -> pd.DataFrame:
    """Compare every gate output net with its logic value, and measure how far it sits from its rail."""
    expected = pd.DataFrame([{net: cla_logic(vector)[net] for net in CLA_NETS} for vector in range(VECTOR_COUNT)])
    expected_levels = expected.loc[simulations["vector"]].to_numpy()
    voltages = simulations[[f"v_{net}" for net in CLA_NETS]].to_numpy()
    vdd = simulations["vdd"].to_numpy()[:, None]
    mismatches = (voltages > vdd / 2).astype(int) != expected_levels
    # Distance from the ideal rail as a fraction of VDD.
    deviation = np.abs(voltages - expected_levels * vdd) / vdd
    return pd.DataFrame([{
        "simulations": len(simulations),
        "net_checks": mismatches.size,
        "mismatches": int(mismatches.sum()),
        "median_rail_deviation_fraction_of_vdd": np.median(deviation),
        "p99_rail_deviation_fraction_of_vdd": np.percentile(deviation, 99),
        "max_rail_deviation_fraction_of_vdd": deviation.max(),
        "max_deviation_net": CLA_NETS[int(np.unravel_index(deviation.argmax(), deviation.shape)[1])],
    }])


def case_counts() -> np.ndarray:
    """Number of adder instances in each gate and input case, per vector: shape (vectors, 18 cases)."""
    case_index = {case: position for position, case in enumerate(GATE_CASES)}
    counts = np.zeros((VECTOR_COUNT, len(GATE_CASES)))
    for vector in range(VECTOR_COUNT):
        for case in cla_gate_cases(vector):
            counts[vector, case_index[case]] += 1
    return counts


def gate_leakage_matrix(predictions: pd.DataFrame, column: str, corners: np.ndarray) -> np.ndarray:
    """Gate leakage per gate and input case and corner: shape (18 cases, corners)."""
    table = predictions.pivot(index=["gate", "inputs"], columns="corner", values=column)
    table = table.reindex(index=pd.MultiIndex.from_tuples(GATE_CASES), columns=corners)
    if table.isna().any().any():
        raise ValueError(f"{column}: {table.isna().sum().sum()} gate cases and corners missing from the Part B results")
    return table.to_numpy()


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    simulations = pd.read_csv(DATA_DIR / "cla_sims.csv")
    predictions = pd.read_csv(DATA_DIR / f"gate_predictions_{name}.csv", dtype={"inputs": str})
    RESULTS_DIR.mkdir(exist_ok=True)

    logic = net_logic_check(simulations)
    logic.to_csv(RESULTS_DIR / "cla_net_logic_check.csv", index=False, float_format="%.6e")
    print(logic.to_string(index=False))
    if logic["mismatches"].iloc[0] != 0:
        raise RuntimeError("SPICE adder nets disagree with the logic evaluation, see results/cla_net_logic_check.csv")

    corners = np.sort(simulations["corner"].unique())
    counts = case_counts()
    composed = {column: counts @ gate_leakage_matrix(predictions, column, corners)
                for column in ("leak_spice", "leak_predicted", "leak_terminal")}
    vector_grid, corner_grid = np.meshgrid(np.arange(VECTOR_COUNT), corners, indexing="ij")
    composed_frame = pd.DataFrame({"vector": vector_grid.ravel(), "corner": corner_grid.ravel(),
                                   "leak_gate_spice_sum": composed["leak_spice"].ravel(),
                                   "leak_predicted": composed["leak_predicted"].ravel(),
                                   "leak_terminal": composed["leak_terminal"].ravel()})
    results = simulations[["vector", "corner", "leak_device"]].rename(columns={"leak_device": "leak_spice"}).merge(
        composed_frame, on=["vector", "corner"], how="left", validate="one_to_one")
    results.to_csv(DATA_DIR / f"cla_predictions_{name}.csv", index=False, float_format="%.12e")

    control = pd.DataFrame([comparison_metrics(results["leak_spice"], results["leak_gate_spice_sum"])])
    control.to_csv(RESULTS_DIR / "cla_composition_control.csv", index=False, float_format="%.6g")
    print("composition control, Part B SPICE gate sum against the SPICE adder")
    print(control.to_string(index=False))

    for column, suffix, label in [("leak_predicted", "", "Kirchhoff"), ("leak_terminal", "_terminal", "per terminal")]:
        overall = pd.DataFrame([comparison_metrics(results["leak_spice"], results[column])])
        overall.to_csv(RESULTS_DIR / f"cla_results_{name}{suffix}.csv", index=False, float_format="%.6g")
        print(f"{name} against the SPICE adder, gates composed {label}")
        print(overall.to_string(index=False))
    by_vector = pd.DataFrame([{"vector": vector, **comparison_metrics(group["leak_spice"], group["leak_predicted"])}
                              for vector, group in results.groupby("vector")])
    by_vector.to_csv(RESULTS_DIR / f"cla_results_by_vector_{name}.csv", index=False, float_format="%.6g")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    parity_plot({"cla": (results["leak_spice"].to_numpy(), results["leak_predicted"].to_numpy())},
                {"cla": "#2a78d6"}, "SPICE adder leakage (A)", f"{name.upper()} composed adder leakage (A)",
                FIGURE_DIR / f"cla_parity_{name}")


if __name__ == "__main__":
    main()
