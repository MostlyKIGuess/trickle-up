"""Predict the four terminal currents of one transistor with the models saved by 04_train.py.

Usage from another script in scripts/:

    from fet_model import load_models
    models = load_models(MODELS_DIR)
    currents = models.predict("nmos_off", frame)

frame has the input columns vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj (volts, degrees
C, metres, and the process values as in the model file). The result has columns id, ig, is, ib in
amps, positive into the terminal.

The input features and the target transforms are defined here, so training and prediction use the
same code. Prediction needs numpy, scikit-learn and LightGBM only. The MLP is stored as numpy
weights, so it runs without PyTorch.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import CASES, INPUT_COLUMNS, TERMINALS

BOLTZMANN_OVER_CHARGE = 8.617333262e-5  # V/K
KELVIN_OFFSET = 273.15
# |Vd - Vs| is floored here before its log is taken. The data has no |Vds| this small.
VDS_FLOOR = 1e-9
# log10|I| floors the current here. With every terminal of an OFF device at 0 V its gate current can
# come out as exactly zero, and anything this small is far below any leakage the gates add up.
CURRENT_FLOOR = 1e-30

# LightGBM records generic feature names when fitted on a numpy array, and scikit-learn then warns
# on every predict with a numpy array. Fitting and prediction both use the FEATURE_NAMES columns.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

FEATURE_NAMES = [
    "vdd", "vd", "vs", "vds", "vds_abs", "log_drain_factor", "vgs_eff", "vbs_eff", "vgd", "vgs", "vdb",
    "vsb", "inverse_temp", "overdrive_over_vt", "log_w", "vth0_abs", "toxe", "ndep", "u0", "xj",
]


def effective_width(frame: pd.DataFrame, width_offset: float) -> np.ndarray:
    """BSIM4's effective channel width W - 2 * wint, with wint from the model file (common.channel_width_offsets)."""
    width = frame["w"].to_numpy(dtype=float) - 2.0 * width_offset
    if np.any(width <= 0):
        raise ValueError(f"effective width not positive for wint = {width_offset}")
    return width


def features(frame: pd.DataFrame, device: str, width_offset: float | None = None) -> np.ndarray:
    """Model inputs of one transistor, one row per row of frame, columns in FEATURE_NAMES order.

    Voltages are signed so that they are positive in the usual operating direction of the device
    (multiplied by -1 for PMOS). BSIM4 is symmetric in drain and source, so the effective source is
    the lower of Vd and Vs for NMOS and the higher for PMOS. With width_offset (wint) given, a last
    column log_weff, the log of the effective width, follows the FEATURE_NAMES columns.
    """
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"input frame lacks columns {missing}")
    if device not in ("nmos", "pmos"):
        raise ValueError(f"device must be nmos or pmos, got {device}")
    polarity = 1.0 if device == "nmos" else -1.0
    vd, vs, vg, vb = (frame[column].to_numpy(dtype=float) for column in ("vd", "vs", "vg", "vb"))
    kelvin = frame["temp"].to_numpy(dtype=float) + KELVIN_OFFSET
    thermal_voltage = BOLTZMANN_OVER_CHARGE * kelvin
    source = np.minimum(vd, vs) if device == "nmos" else np.maximum(vd, vs)
    vds_abs = np.abs(vd - vs)
    vgs_eff = polarity * (vg - source)
    columns = {
        "vdd": frame["vdd"].to_numpy(dtype=float),
        "vd": vd,
        "vs": vs,
        "vds": polarity * (vd - vs),
        "vds_abs": vds_abs,
        # The subthreshold channel current scales as 1 - exp(-|Vds| / vT), this is its log.
        "log_drain_factor": np.log(-np.expm1(-np.maximum(vds_abs, VDS_FLOOR) / thermal_voltage)),
        "vgs_eff": vgs_eff,
        "vbs_eff": polarity * (vb - source),
        "vgd": polarity * (vg - vd),
        "vgs": polarity * (vg - vs),
        "vdb": polarity * (vd - vb),
        "vsb": polarity * (vs - vb),
        "inverse_temp": 1.0 / kelvin,
        # The exponent of the subthreshold current, (Vgs - Vth) / vT.
        "overdrive_over_vt": (vgs_eff - np.abs(frame["vth0"].to_numpy(dtype=float))) / thermal_voltage,
        "log_w": np.log(frame["w"].to_numpy(dtype=float)),
        "vth0_abs": np.abs(frame["vth0"].to_numpy(dtype=float)),
        "toxe": frame["toxe"].to_numpy(dtype=float),
        "ndep": frame["ndep"].to_numpy(dtype=float),
        "u0": frame["u0"].to_numpy(dtype=float),
        "xj": frame["xj"].to_numpy(dtype=float),
    }
    names = list(FEATURE_NAMES)
    if width_offset is not None:
        columns["log_weff"] = np.log(effective_width(frame, width_offset))
        names.append("log_weff")
    return np.column_stack([columns[name] for name in names])


def width_divisor(frame: pd.DataFrame, width_offset: float | None, scaled_terminals: list[int]) -> np.ndarray:
    """Per row and terminal, the effective width for the terminals whose target is a current per width, else 1."""
    divisor = np.ones((len(frame), len(TERMINALS)))
    if scaled_terminals:
        divisor[:, scaled_terminals] = effective_width(frame, width_offset)[:, None]
    return divisor


def to_target(currents: np.ndarray, transform: str, current_scale: np.ndarray,
              divisor: np.ndarray | None = None) -> np.ndarray:
    """Currents (rows, 4 terminals) to the regression target of the transform.

    divisor (rows, 4), from width_divisor, turns a current into a current per effective width first.
    Only the log transform takes it, since current_scale of asinh is in amps.
    """
    if divisor is not None and transform != "log":
        raise ValueError("a width divisor needs the log transform")
    if transform == "log":
        scaled = currents if divisor is None else currents / divisor
        return np.log10(np.maximum(np.abs(scaled), CURRENT_FLOOR))
    if transform == "asinh":
        return np.arcsinh(currents / current_scale)
    raise ValueError(f"unknown transform {transform}")


def from_target(targets: np.ndarray, transform: str, current_scale: np.ndarray, signs: np.ndarray,
                divisor: np.ndarray | None = None) -> np.ndarray:
    """Inverse of to_target. signs (+1 or -1 per entry) is used by the log transform only."""
    if transform == "log":
        currents = signs * 10.0 ** targets
        return currents if divisor is None else currents * divisor
    if transform == "asinh":
        return np.sinh(targets) * current_scale
    raise ValueError(f"unknown transform {transform}")


def silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


def run_network(entry: dict, inputs: np.ndarray) -> np.ndarray:
    """Outputs of a fitted model entry, as written by 04_train.py, on standardized inputs."""
    if entry["kind"] == "sklearn":
        outputs = np.asarray(entry["estimator"].predict(inputs), dtype=float)
        return outputs.reshape(len(inputs), -1)
    if entry["kind"] == "mlp":
        hidden = inputs
        last = len(entry["weights"]) - 1
        for index, (weight, bias) in enumerate(zip(entry["weights"], entry["biases"])):
            hidden = hidden @ weight + bias
            if index < last:
                hidden = silu(hidden)
        return hidden
    raise ValueError(f"unknown model entry kind {entry['kind']}")


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Terminal currents from one saved case bundle. 04_train.py evaluates through this as well.

    Bundles saved before the options of 04_train.py existed lack their keys and predict as before:
    - width_offset, width_scaled_terminals: targets of those terminals are currents per effective width
    - separate: a second regressor whose outputs replace the targets of its terminals
    - kind "ensemble": members of the same transform, averaged in log10|I| with the majority sign
    """
    if bundle.get("kind") == "ensemble":
        return predict_ensemble(bundle, frame)
    width_offset = bundle.get("width_offset")
    inputs = (features(frame, bundle["device"], width_offset) - bundle["feature_mean"]) / bundle["feature_std"]
    targets = run_network(bundle["regressor"], inputs) * bundle["target_std"] + bundle["target_mean"]
    separate = bundle.get("separate")
    if separate is not None:
        targets[:, separate["terminals"]] = (run_network(separate["regressor"], inputs) * separate["target_std"]
                                             + separate["target_mean"])
    signs = np.tile(bundle["constant_signs"], (len(frame), 1)).astype(float)
    if bundle["transform"] == "log" and bundle["mixed_terminals"]:
        # The classifier of the MLP returns logits, the scikit-learn classifiers return 0 or 1.
        threshold = 0.0 if bundle["sign_classifier"]["kind"] == "mlp" else 0.5
        positive = run_network(bundle["sign_classifier"], inputs) > threshold
        signs[:, bundle["mixed_terminals"]] = np.where(positive, 1.0, -1.0)
    scaled_terminals = bundle.get("width_scaled_terminals", [])
    divisor = width_divisor(frame, width_offset, scaled_terminals) if scaled_terminals else None
    currents = from_target(targets, bundle["transform"], bundle["current_scale"], signs, divisor)
    return pd.DataFrame(currents, columns=TERMINALS, index=frame.index)


def predict_ensemble(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Geometric mean of the members' current magnitudes, with the sign most members give."""
    members = [predict_bundle(member, frame).to_numpy() for member in bundle["members"]]
    if len(members) % 2 == 0:
        raise ValueError("an ensemble needs an odd number of members so the sign vote cannot tie")
    stacked = np.stack(members)
    magnitude = 10.0 ** np.log10(np.maximum(np.abs(stacked), CURRENT_FLOOR)).mean(axis=0)
    signs = np.where(np.sign(stacked).sum(axis=0) >= 0, 1.0, -1.0)
    return pd.DataFrame(signs * magnitude, columns=TERMINALS, index=frame.index)


class FetModels:
    """The four chosen transistor models, one per case."""

    def __init__(self, bundles: dict):
        self.bundles = bundles

    def predict(self, case_name: str, frame: pd.DataFrame) -> pd.DataFrame:
        if case_name not in self.bundles:
            raise KeyError(f"no model for case {case_name}, have {sorted(self.bundles)}")
        return predict_bundle(self.bundles[case_name], frame)


def load_models(models_dir: Path) -> FetModels:
    models_dir = Path(models_dir)
    bundles = {}
    for case_name in CASES:
        path = models_dir / f"fet_{case_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing, run scripts/04_train.py first")
        bundles[case_name] = joblib.load(path)
    return FetModels(bundles)
