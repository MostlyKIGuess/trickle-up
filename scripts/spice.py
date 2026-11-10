"""Model cards with per-sample process values, and a runner that returns what ngspice printed."""

import re
import subprocess
import tempfile
from pathlib import Path

from common import MODEL_FILE, PROCESS_PARAMETERS

_MODEL_TEXT = MODEL_FILE.read_text()
_PRINTED_VALUE = re.compile(r"^(\w+)\s*=\s*([-+0-9.eE]+)\s*$")


def model_cards(process_values):
    """Return the model file text with vth0, toxe, ndep, u0 and xj replaced.

    process_values maps "nmos" and/or "pmos" to {parameter: value}. A device left out keeps its
    nominal values.
    """
    blocks = re.split(r"(?=^\.model)", _MODEL_TEXT, flags=re.MULTILINE)
    edited = []
    for block in blocks:
        header = re.match(r"\.model\s+(\w+)", block)
        if header is not None and header.group(1) in process_values:
            for parameter in PROCESS_PARAMETERS:
                value = process_values[header.group(1)][parameter]
                block, count = re.subn(rf"\b({parameter}\s*=\s*)[-+0-9.eE]+", rf"\g<1>{value:.10e}", block)
                if count != 1:
                    raise ValueError(f"{parameter} found {count} times in the {header.group(1)} model")
        edited.append(block)
    return "".join(edited)


def run_deck(deck_text, expected_names):
    """Run one deck in batch mode and return the printed values named in expected_names."""
    with tempfile.TemporaryDirectory() as work_dir:
        deck_path = Path(work_dir) / "deck.cir"
        deck_path.write_text(deck_text)
        result = subprocess.run(["ngspice", "-b", str(deck_path)], capture_output=True, text=True,
                                cwd=work_dir, timeout=120)
    values = {}
    for line in result.stdout.splitlines():
        match = _PRINTED_VALUE.match(line.strip())
        if match is not None:
            values[match.group(1).lower()] = float(match.group(2))
    missing = [name for name in expected_names if name not in values]
    if result.returncode != 0 or missing:
        raise RuntimeError(f"ngspice exit {result.returncode}, missing {missing}\n"
                           f"--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-3000:]}")
    return {name: values[name] for name in expected_names}


def ngspice_version():
    result = subprocess.run(["ngspice", "-v"], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if "ngspice-" in line:
            return line.strip("* ").strip()
    raise RuntimeError(f"no version line in ngspice -v output:\n{result.stdout}")
