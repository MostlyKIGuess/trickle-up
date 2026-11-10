"""Settings shared by every script: the team's assignment, the sweep ranges and the paths."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "given" / "models" / "22nm_LP.pm"
DATA_DIR = ROOT / "data"
SPICE_DIR = ROOT / "spice"

# Team "Dvd Player" assignment. The PTM file states 0.95 V as its nominal supply, the TAs assigned 1 V.
VDD_NOMINAL = 1.0
L_MIN = 22e-9
# Diffusion length from the TA example netlist (.PARAM Ldiff=44n). Without AD, AS, PD and PS the
# BSIM4 drain junction current sits at 1e-14 A whatever the width and temperature.
L_DIFF = 44e-9

TEMP_RANGE = (-40.0, 125.0)
VDD_SPREAD = 0.10
WIDTH_RANGE = (1 * L_MIN, 8 * L_MIN)
PROCESS_SPREAD = 0.10
PROCESS_PARAMETERS = ["vth0", "toxe", "ndep", "u0", "xj"]

# Every netlist carries these, as the brief requires.
SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"

# Gate and body voltages fixed by each case, as fractions of VDD.
CASES = {
    "nmos_on": {"device": "nmos", "gate": 1.0, "body": 0.0},
    "nmos_off": {"device": "nmos", "gate": 0.0, "body": 0.0},
    "pmos_on": {"device": "pmos", "gate": 0.0, "body": 1.0},
    "pmos_off": {"device": "pmos", "gate": 1.0, "body": 1.0},
}

DATA_COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROCESS_PARAMETERS, "id", "ig", "is", "ib"]

# File name suffix of each sample set: the uniform set of the brief, the targeted set of
# 01_sample_fet.py with drain and source near each other or at the rails, and the gate-bias set with
# the terminal voltages a transistor has inside a static gate.
SAMPLE_SETS = {"uniform": "", "targeted": "_targeted", "gatebias": "_gatebias"}
# The sets 04_train.py fits on unless told otherwise. Every set is evaluated.
DEFAULT_TRAINING_SETS = ["uniform", "targeted"]


def diffusion_geometry(width):
    """Instance parameters for the source and drain diffusions of a transistor of this width."""
    area = width * L_DIFF
    perimeter = 2 * (width + L_DIFF)
    return f"AD={area:.6e} AS={area:.6e} PD={perimeter:.6e} PS={perimeter:.6e}"


def nominal_process_values():
    """Read vth0, toxe, ndep, u0 and xj of both models from the model file."""
    return model_card_values(PROCESS_PARAMETERS)


def channel_width_offsets() -> dict[str, float]:
    """wint of each device from the model file. BSIM4's effective width is W - 2 * wint."""
    return {device: values["wint"] for device, values in model_card_values(["wint"]).items()}


def model_card_values(parameters: list[str]) -> dict[str, dict[str, float]]:
    """Read the named parameters of both models from the model file, keyed by device then parameter."""
    text = MODEL_FILE.read_text()
    blocks = re.split(r"(?=^\.model)", text, flags=re.MULTILINE)
    values = {}
    for block in blocks:
        header = re.match(r"\.model\s+(\w+)", block)
        if header is None:
            continue
        device = header.group(1)
        values[device] = {}
        for parameter in parameters:
            found = re.search(rf"\b{parameter}\s*=\s*([-+0-9.eE]+)", block)
            if found is None:
                raise ValueError(f"{parameter} missing from the {device} model in {MODEL_FILE}")
            values[device][parameter] = float(found.group(1))
    if set(values) != {"nmos", "pmos"}:
        raise ValueError(f"expected nmos and pmos models in {MODEL_FILE}, found {sorted(values)}")
    return values


RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
# Parts B and C share these corners, each one draw of VDD, temperature and one process set per device.
CORNER_COUNT = 200
# The model cards state tnom = 27, the nominal corner of the CLA logic check.
TEMP_NOMINAL = 27.0
# ngspice workers. The GPU machine has 16 cores and the training run uses up to 8 of them.
SPICE_WORKERS = 8

# The transistor models: their inputs, their outputs and where they are saved.
INPUT_COLUMNS = DATA_COLUMNS[:12]
TERMINALS = ["id", "ig", "is", "ib"]
MODELS_DIR = ROOT / "models"
