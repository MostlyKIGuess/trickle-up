"""Gate leakage composed from per-transistor current predictions, against the SPICE gate leakage.

Usage: python3 scripts/07_gates_ml.py oracle
       python3 scripts/07_gates_ml.py <models_dir>

Each transistor of each gate simulation gets one model input row: the corner's vdd, temperature and
the process set of its device type, its width, and its terminal voltages from the SPICE-recorded node
voltages. Its gate voltage picks the case (ON or OFF). The terminal currents on nodes held at VDD
(the supply and every input at 1) sum to the gate leakage.

Two compositions:
  Kirchhoff (leak_predicted, the method)  every OFF transistor's currents and every gate and body
      current as predicted, the drain and source currents of ON transistors from current balance at
      the free nodes (balanced_currents)
  per terminal (leak_terminal, baseline)  every predicted terminal current as it is. ON transistors
      sit at |Vds| of microvolts, where a prediction of the channel current is ill-conditioned, and
      this misses gates whose leakage leaves VDD through an ON PMOS.

"oracle" stands in for the transistor models with each transistor's SPICE terminal currents. Both
compositions must then reproduce the device-based SPICE gate leakage to about reltol, which checks
the row building, the case split and the composition independently of any model. With a models
directory, the transistor models come from fet_model.load_models.

Writes, with <name> = oracle or ml,
  data/gate_predictions_<name>.csv                     per gate simulation: SPICE and both compositions
  results/gate_results_<name>.csv                      Kirchhoff, per gate and input case and overall
  results/gate_results_<name>_terminal.csv             per terminal, the same
  results/transistor_r2_<name>.csv                     R2 of log10|I| of the currents the composition uses
  results/gate_worst_<name>.csv                        the largest errors and the current behind each
  results/figures/gate_parity_<name>[_terminal].{pdf,png}
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import CASES, DATA_DIR, FIGURE_DIR, PROCESS_PARAMETERS, RESULTS_DIR
from gates import (CURRENT_COLUMNS, GATE_CASES, GATE_COLORS, GATES, GROUND, MODEL_INPUT_COLUMNS, SUPPLY, TERMINALS,
                   comparison_metrics, gate_nodes_held_high, parity_plot, terminal_weights)

KEY = ["gate", "inputs", "corner"]
# The OFF Ig floor of 04_train.py --mask-off-ig. Transistor R2 leaves out currents at or below it.
COMPOSITION_R2_FLOOR = 1e-26
# Rows of results/gate_worst_<name>.csv.
WORST_CASE_COUNT = 20


class OracleModels:
    """Returns the SPICE-recorded terminal currents of the rows it is asked about."""

    def __init__(self, transistors: pd.DataFrame):
        self.transistors = transistors

    def predict(self, case_name: str, frame: pd.DataFrame) -> pd.DataFrame:
        recorded = self.transistors.loc[frame.index]
        wrong_device = recorded["device"] != CASES[case_name]["device"]
        if wrong_device.any():
            raise ValueError(f"{wrong_device.sum()} {recorded['device'][wrong_device].iloc[0]} rows sent to {case_name}")
        return recorded[CURRENT_COLUMNS].reset_index(drop=True)


def model_inputs(transistors: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """One transistor-model input row per transistor row, on the same index."""
    corner_columns = corners.set_index("corner").loc[transistors["corner"]].reset_index(drop=True)
    corner_columns.index = transistors.index
    frame = transistors[["w", "vd", "vs", "vg", "vb"]].copy()
    frame["vdd"] = corner_columns["vdd"]
    frame["temp"] = corner_columns["temp"]
    is_nmos = transistors["device"] == "nmos"
    for parameter in PROCESS_PARAMETERS:
        frame[parameter] = np.where(is_nmos, corner_columns[f"nmos_{parameter}"], corner_columns[f"pmos_{parameter}"])
    return frame[MODEL_INPUT_COLUMNS]


def transistor_cases(transistors: pd.DataFrame, inputs: pd.DataFrame) -> pd.Series:
    """ON or OFF from the gate voltage: an NMOS is ON with its gate high, a PMOS with its gate low."""
    gate_high = inputs["vg"] > inputs["vdd"] / 2
    is_nmos = transistors["device"] == "nmos"
    on = np.where(is_nmos, gate_high, ~gate_high)
    return pd.Series(np.where(is_nmos, np.where(on, "nmos_on", "nmos_off"), np.where(on, "pmos_on", "pmos_off")),
                     index=transistors.index)


def predict_currents(models, inputs: pd.DataFrame, cases: pd.Series) -> pd.DataFrame:
    currents = pd.DataFrame(np.nan, index=inputs.index, columns=CURRENT_COLUMNS)
    for case_name in CASES:
        selected = inputs[cases == case_name]
        if selected.empty:
            continue
        predicted = models.predict(case_name, selected)
        if len(predicted) != len(selected) or list(predicted.columns) != CURRENT_COLUMNS:
            raise ValueError(f"{case_name}: expected {len(selected)} rows with columns {CURRENT_COLUMNS}, "
                             f"got {len(predicted)} rows with {list(predicted.columns)}")
        currents.loc[selected.index] = predicted.to_numpy()
    if currents.isna().any().any():
        raise ValueError(f"{currents.isna().any(axis=1).sum()} transistor rows without predicted currents")
    return currents


def composition_weights() -> pd.DataFrame:
    frames = []
    for gate_name, inputs in GATE_CASES:
        weights = terminal_weights(GATES[gate_name].transistors, gate_nodes_held_high(gate_name, inputs))
        weights = weights.add_prefix("weight_").rename_axis("transistor").reset_index()
        weights["gate"], weights["inputs"] = gate_name, inputs
        frames.append(weights)
    return pd.concat(frames, ignore_index=True)


def compose_gate_leakage(transistors: pd.DataFrame, currents: pd.DataFrame) -> pd.Series:
    weighted = transistors[KEY + ["transistor"]].merge(composition_weights(), on=["gate", "inputs", "transistor"],
                                                       how="left", validate="many_to_one")
    weight_columns = [f"weight_{current}" for current in CURRENT_COLUMNS]
    if weighted[weight_columns].isna().any().any():
        raise ValueError("transistor rows without composition weights")
    contribution = (weighted[weight_columns].to_numpy() * currents.to_numpy()).sum(axis=1)
    return pd.Series(contribution, index=pd.MultiIndex.from_frame(weighted[KEY])).groupby(level=KEY).sum()


def balanced_currents(transistors: pd.DataFrame, currents: pd.DataFrame, cases: pd.Series) -> pd.DataFrame:
    """Currents with the drain and source currents of every ON transistor replaced by current balance.

    An ON transistor inside a gate has microvolts across it and carries whatever the OFF transistors
    and gate terminals on its free node draw, so its drain current follows from Kirchhoff's current law
    at the free nodes (output and stack nodes) far better than from its own bias. Its gate and body
    currents stay as predicted, and its source current is -(id + ig + ib). Each gate and input case
    has the same ON transistors at every corner, so one linear system per case is solved for all corners.
    Parallel ON transistors share a node pair, the least-squares solution splits their current equally,
    and only their sum reaches the leakage.
    """
    balanced = currents.copy()
    is_on = cases.str.endswith("_on")
    for (gate_name, inputs), rows in transistors.groupby(["gate", "inputs"], sort=False):
        gate = GATES[gate_name]
        fixed = {SUPPLY, GROUND, *gate.inputs}
        free = sorted({transistor.node(terminal) for transistor in gate.transistors for terminal in TERMINALS} - fixed)
        # Row labels of each transistor, in corner order.
        labels = {name: group.sort_values("corner").index for name, group in rows.groupby("transistor")}
        on_names = []
        for transistor in gate.transistors:
            states = is_on.loc[labels[transistor.name]]
            if states.any() and not states.all():
                raise ValueError(f"{gate_name} {inputs} {transistor.name} is ON at some corners and OFF at others")
            if states.all():
                on_names.append(transistor.name)
        if not on_names or not free:
            continue
        corner_count = len(labels[on_names[0]])
        coefficients = np.zeros((len(free), len(on_names)))
        known = np.zeros((len(free), corner_count))
        for row, node in enumerate(free):
            for transistor in gate.transistors:
                terminal_currents = currents.loc[labels[transistor.name]]
                for terminal, column in zip(TERMINALS, CURRENT_COLUMNS):
                    if transistor.node(terminal) != node:
                        continue
                    if transistor.name in on_names and terminal in ("drain", "source"):
                        unknown = on_names.index(transistor.name)
                        if terminal == "drain":
                            coefficients[row, unknown] += 1
                        else:
                            coefficients[row, unknown] -= 1
                            known[row] += (terminal_currents["ig"] + terminal_currents["ib"]).to_numpy()
                    else:
                        known[row] -= terminal_currents[column].to_numpy()
        drain = np.linalg.lstsq(coefficients, known, rcond=None)[0]
        for unknown, name in enumerate(on_names):
            # Drain and source on the same node, or neither on a free node: no balance constrains it.
            if not coefficients[:, unknown].any():
                continue
            label = labels[name]
            balanced.loc[label, "id"] = drain[unknown]
            balanced.loc[label, "is"] = -(currents.loc[label, "ig"] + currents.loc[label, "ib"]).to_numpy() - drain[unknown]
    return balanced


def transistor_r2(transistors: pd.DataFrame, currents: pd.DataFrame, cases: pd.Series) -> pd.DataFrame:
    """R2 of log10|I| for the transistor currents the Kirchhoff composition takes from the models.

    Those are all four currents of OFF transistors and the gate and body currents of ON transistors.
    Currents with |I| at or below COMPOSITION_R2_FLOOR in SPICE are left out and counted, since their
    log is unbounded and they are seven decades below any gate leakage. sign_agreement is the fraction
    of the kept currents whose predicted sign matches SPICE.
    """
    rows = []
    groups = [(case_name, column) for case_name in CASES for column in CURRENT_COLUMNS
              if case_name.endswith("_off") or column in ("ig", "ib")]
    for case_name, column in [*groups, ("all", "all")]:
        if case_name == "all":
            spice = np.concatenate([transistors.loc[cases == name, current].to_numpy() for name, current in groups])
            predicted = np.concatenate([currents.loc[cases == name, current].to_numpy() for name, current in groups])
        else:
            spice = transistors.loc[cases == case_name, column].to_numpy()
            predicted = currents.loc[cases == case_name, column].to_numpy()
        kept = np.abs(spice) > COMPOSITION_R2_FLOOR
        spice_log = np.log10(np.abs(spice[kept]))
        predicted_log = np.log10(np.maximum(np.abs(predicted[kept]), np.finfo(float).tiny))
        residual = ((spice_log - predicted_log) ** 2).sum()
        total = ((spice_log - spice_log.mean()) ** 2).sum()
        rows.append({"case": case_name, "current": column, "rows": len(spice), "below_floor": int((~kept).sum()),
                     "r2_log10": 1 - residual / total,
                     "median_abs_log10_error": np.median(np.abs(spice_log - predicted_log)),
                     "sign_agreement": np.mean(np.sign(spice[kept]) == np.sign(predicted[kept]))})
    return pd.DataFrame(rows)


def worst_cases(predictions: pd.DataFrame, transistors: pd.DataFrame, corners: pd.DataFrame,
                currents: pd.DataFrame, cases: pd.Series, count: int) -> pd.DataFrame:
    """The gate simulations with the largest Kirchhoff error, and the predicted current behind each.

    The Kirchhoff leakage is linear in the predicted currents, so its error is the sum, over every
    current the composition takes from the models, of that current's error carried through the
    current balance. top_share is the largest term over the whole error.
    """
    error = (predictions["leak_predicted"] - predictions["leak_spice"]) / predictions["leak_spice"]
    worst = predictions.assign(error_percent=100 * error).reindex(
        error.abs().sort_values(ascending=False).index[:count])
    current_errors = currents - transistors[CURRENT_COLUMNS]
    rows = []
    for record in worst.to_dict("records"):
        selected = transistors.index[(transistors["gate"] == record["gate"]) & (transistors["inputs"] == record["inputs"])
                                     & (transistors["corner"] == record["corner"])]
        terms = []
        for label in selected:
            for column in CURRENT_COLUMNS:
                if cases[label].endswith("_on") and column in ("id", "is"):
                    continue
                single = pd.DataFrame(0.0, index=selected, columns=CURRENT_COLUMNS)
                single.loc[label, column] = current_errors.loc[label, column]
                carried = compose_gate_leakage(transistors.loc[selected],
                                               balanced_currents(transistors.loc[selected], single, cases[selected]))
                terms.append((transistors.loc[label, "transistor"], cases[label], column, carried.iloc[0]))
        top = max(terms, key=lambda term: abs(term[3]))
        total = record["leak_predicted"] - record["leak_spice"]
        corner = corners.set_index("corner").loc[record["corner"]]
        rows.append({**{key: record[key] for key in KEY}, "vdd": corner["vdd"], "temp": corner["temp"],
                     "leak_spice_A": record["leak_spice"], "leak_predicted_A": record["leak_predicted"],
                     "error_percent": record["error_percent"], "top_transistor": top[0], "top_case": top[1],
                     "top_current": top[2], "top_share": top[3] / total,
                     "terms_over_error": sum(term[3] for term in terms) / total})
    return pd.DataFrame(rows)


def result_table(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [{"gate": gate_name, "inputs": inputs, **comparison_metrics(group["leak_spice"], group[column])}
            for (gate_name, inputs), group in predictions.groupby(["gate", "inputs"], sort=False)]
    rows.append({"gate": "all", "inputs": "all", **comparison_metrics(predictions["leak_spice"], predictions[column])})
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    transistors = pd.read_csv(DATA_DIR / "gate_transistors.csv", dtype={"inputs": str})
    simulations = pd.read_csv(DATA_DIR / "gate_sims.csv", dtype={"inputs": str})
    corners = pd.read_csv(DATA_DIR / "corners.csv")
    if sys.argv[1] == "oracle":
        name, models = "oracle", OracleModels(transistors)
    else:
        from fet_model import load_models
        name, models = "ml", load_models(Path(sys.argv[1]))

    inputs = model_inputs(transistors, corners)
    cases = transistor_cases(transistors, inputs)
    currents = predict_currents(models, inputs, cases)
    balanced = balanced_currents(transistors, currents, cases)
    composed = pd.concat([
        compose_gate_leakage(transistors, balanced).rename("leak_predicted"),
        compose_gate_leakage(transistors, currents).rename("leak_terminal"),
    ], axis=1).reset_index()

    predictions = simulations[KEY + ["leak_device"]].rename(columns={"leak_device": "leak_spice"}).merge(
        composed, on=KEY, how="left", validate="one_to_one")
    if predictions[["leak_predicted", "leak_terminal"]].isna().any().any():
        raise ValueError("gate simulations without a composed leakage")
    predictions.to_csv(DATA_DIR / f"gate_predictions_{name}.csv", index=False, float_format="%.12e")

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    r2 = transistor_r2(transistors, currents, cases)
    r2.to_csv(RESULTS_DIR / f"transistor_r2_{name}.csv", index=False, float_format="%.6g")
    print(f"{name}, transistor R2 on log10|I|, currents the Kirchhoff composition takes from the models")
    print(r2.to_string(index=False))
    worst = worst_cases(predictions, transistors, corners, currents, cases, WORST_CASE_COUNT)
    worst.to_csv(RESULTS_DIR / f"gate_worst_{name}.csv", index=False, float_format="%.6g")
    print(f"{name}, the {WORST_CASE_COUNT} largest Kirchhoff errors")
    print(worst.to_string(index=False))

    for column, suffix, label in [("leak_predicted", "", "Kirchhoff"), ("leak_terminal", "_terminal", "per terminal")]:
        table = result_table(predictions, column)
        table.to_csv(RESULTS_DIR / f"gate_results_{name}{suffix}.csv", index=False, float_format="%.6g")
        print(f"{name}, ON drain and source currents {label}")
        print(table.to_string(index=False))
        groups = {gate_name: (group["leak_spice"].to_numpy(), group[column].to_numpy())
                  for gate_name, group in predictions.groupby("gate", sort=False)}
        parity_plot(groups, GATE_COLORS, "SPICE gate leakage (A)", f"{name.upper()} gate leakage (A)",
                    FIGURE_DIR / f"gate_parity_{name}{suffix}")


if __name__ == "__main__":
    main()
