"""Time of one adder leakage with SPICE and with our method, in a single process.

Usage: python3 scripts/13_timing.py [corners] [vectors_per_corner] [data_dir]

SPICE needs one ngspice run of the adder (204 transistors) per vector and corner. Our method needs the
18 gate cases of a corner in ngspice, for the node voltages the transistor models take as input, and
then the models, the Kirchhoff sum and the sum over the 42 gates. That gives the leakage of all 128
vectors of the corner, so the cost of one adder leakage is the cost per corner divided by 128.

Runs on the first [corners] corners of data_dir/corners.csv (default 10), with [vectors_per_corner]
adder vectors spread over the 128 (default 16). The ML step is timed in one batch over every corner
of the file, which is how 07_gates_ml.py and 09_cla_ml.py run it. The time to simulate the training
samples and to train the models is paid once and is not measured here.

Reads data_dir/corners.csv, cla_sims.csv, gate_transistors.csv, gate_predictions_ml.csv and models/.
Each timed adder run must reproduce cla_sims.csv and the timed prediction gate_predictions_ml.csv.
Writes results/timing.csv.
"""

import importlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, RESULTS_DIR, ROOT
from fet_model import load_models
from gates import GATE_CASES, VECTOR_COUNT

cla_spice = importlib.import_module("08_cla_spice")
gates_spice = importlib.import_module("06_gates_spice")
gates_ml = importlib.import_module("07_gates_ml")
cla_ml = importlib.import_module("09_cla_ml")

DEFAULT_CORNERS = 10
DEFAULT_VECTORS_PER_CORNER = 16
ML_REPEATS = 3
# A timed run must agree with the recorded one to this relative difference.
SPICE_TOLERANCE = 1e-6
PREDICTION_TOLERANCE = 1e-9


def time_adder_spice(corners: list[dict], vectors: np.ndarray, recorded: pd.Series) -> list[float]:
    seconds = []
    for corner in corners:
        for vector in vectors:
            start = time.perf_counter()
            result = cla_spice.simulate_vector((int(vector), corner))
            seconds.append(time.perf_counter() - start)
            expected = recorded.loc[(int(vector), corner["corner"])]
            if abs(result["leak_device"] / expected - 1) > SPICE_TOLERANCE:
                raise RuntimeError(f"adder vector {vector} at corner {corner['corner']} differs from cla_sims.csv")
    return seconds


def time_gate_spice(corners: list[dict]) -> list[float]:
    seconds = []
    for corner in corners:
        for gate_name, inputs in GATE_CASES:
            start = time.perf_counter()
            gates_spice.simulate_gate((gate_name, inputs, corner))
            seconds.append(time.perf_counter() - start)
    return seconds


def predict_adder(models, transistors: pd.DataFrame, corners: pd.DataFrame,
                  counts: np.ndarray) -> tuple[pd.Series, np.ndarray]:
    """Gate leakage per gate case and corner, and the adder leakage per vector and corner."""
    inputs = gates_ml.model_inputs(transistors, corners)
    cases = gates_ml.transistor_cases(transistors, inputs)
    currents = gates_ml.predict_currents(models, inputs, cases)
    balanced = gates_ml.balanced_currents(transistors, currents, cases)
    gate_leakage = gates_ml.compose_gate_leakage(transistors, balanced)
    table = gate_leakage.rename("leak").reset_index().pivot(index=["gate", "inputs"], columns="corner", values="leak")
    table = table.reindex(index=pd.MultiIndex.from_tuples(GATE_CASES))
    return gate_leakage, counts @ table.to_numpy()


def main() -> None:
    corner_count = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORNERS
    vectors_per_corner = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_VECTORS_PER_CORNER
    data_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else DATA_DIR

    corners = pd.read_csv(data_dir / "corners.csv")
    timed_corners = corners.head(corner_count).to_dict("records")
    vectors = np.linspace(0, VECTOR_COUNT - 1, vectors_per_corner).round().astype(int)
    recorded_adder = pd.read_csv(data_dir / "cla_sims.csv").set_index(["vector", "corner"])["leak_device"]
    adder_seconds = time_adder_spice(timed_corners, vectors, recorded_adder)
    gate_seconds = time_gate_spice(timed_corners)

    models = load_models(ROOT / "models")
    transistors = pd.read_csv(data_dir / "gate_transistors.csv", dtype={"inputs": str})
    counts = cla_ml.case_counts()
    ml_seconds = []
    for _ in range(ML_REPEATS):
        start = time.perf_counter()
        gate_leakage, _ = predict_adder(models, transistors, corners, counts)
        ml_seconds.append(time.perf_counter() - start)
    recorded_gates = pd.read_csv(data_dir / "gate_predictions_ml.csv", dtype={"inputs": str})
    check = gate_leakage.rename("timed").reset_index().merge(recorded_gates, on=["gate", "inputs", "corner"],
                                                              validate="one_to_one")
    difference = (check["timed"] / check["leak_predicted"] - 1).abs().max()
    if len(check) != len(recorded_gates) or difference > PREDICTION_TOLERANCE:
        raise RuntimeError(f"the timed prediction differs from gate_predictions_ml.csv by {difference:.3e}")

    adder_run = statistics.mean(adder_seconds)
    gate_run = statistics.mean(gate_seconds)
    ml_per_corner = statistics.median(ml_seconds) / len(corners)
    method_per_corner = len(GATE_CASES) * gate_run + ml_per_corner
    method_per_leakage = method_per_corner / VECTOR_COUNT
    table = pd.DataFrame([
        ("spice_adder_runs", len(adder_seconds)),
        ("spice_adder_seconds_per_run", adder_run),
        ("spice_gate_runs", len(gate_seconds)),
        ("spice_gate_seconds_per_run", gate_run),
        ("ml_corners_in_batch", len(corners)),
        ("ml_seconds_per_corner", ml_per_corner),
        ("method_seconds_per_corner", method_per_corner),
        ("method_seconds_per_adder_leakage", method_per_leakage),
        ("spice_seconds_per_adder_leakage", adder_run),
        ("speedup", adder_run / method_per_leakage),
    ], columns=["name", "value"])
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "timing.csv", index=False, float_format="%.6g")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
