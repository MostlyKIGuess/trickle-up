"""Draw the PVT corners shared by the gate (Part B) and test-circuit (Part C) simulations.

Usage: python3 scripts/05_corners.py [corner_count]

Writes data/corners.csv. Each corner is one draw of VDD, temperature, one NMOS process set and one
PMOS process set, with the distributions of 01_sample_fet.py: VDD uniform within 10 % of nominal,
temperature uniform over -40 to 125 C, and vth0, toxe, ndep, u0 and xj each uniform within 10 % of
the model file's value. All NMOS of a circuit share the NMOS set and all PMOS share the PMOS set.
"""

import sys

import numpy as np
import pandas as pd

from common import (CORNER_COUNT, DATA_DIR, PROCESS_PARAMETERS, PROCESS_SPREAD, TEMP_RANGE, VDD_NOMINAL,
                    VDD_SPREAD, nominal_process_values)

SEED = 463
# 01_sample_fet.py draws streams 0 to 3, one per transistor case. The corners take the next one.
CORNER_STREAM = 4


def draw_corners(corner_count: int, generator: np.random.Generator, nominal: dict) -> pd.DataFrame:
    columns = {
        "corner": np.arange(corner_count),
        "vdd": generator.uniform(VDD_NOMINAL * (1 - VDD_SPREAD), VDD_NOMINAL * (1 + VDD_SPREAD), corner_count),
        "temp": generator.uniform(*TEMP_RANGE, corner_count),
    }
    for device in ("nmos", "pmos"):
        for parameter in PROCESS_PARAMETERS:
            factor = generator.uniform(1 - PROCESS_SPREAD, 1 + PROCESS_SPREAD, corner_count)
            columns[f"{device}_{parameter}"] = nominal[device][parameter] * factor
    return pd.DataFrame(columns)


def main():
    corner_count = int(sys.argv[1]) if len(sys.argv) > 1 else CORNER_COUNT
    generator = np.random.default_rng([SEED, CORNER_STREAM])
    corners = draw_corners(corner_count, generator, nominal_process_values())
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "corners.csv"
    corners.to_csv(path, index=False, float_format="%.10e")
    print(f"{path.relative_to(DATA_DIR.parent)}: {len(corners)} corners")


if __name__ == "__main__":
    main()
