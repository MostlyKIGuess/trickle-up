"""Draw the random input parameters for the four single-transistor cases.

Usage: python3 scripts/01_sample_fet.py [samples_per_case] [targeted_samples_per_case] [gatebias_samples_per_case]

Writes data/samples_<case>.csv with the input columns of the dataset, every input uniform as the
brief specifies. VDD is drawn first because the ranges of Vd and Vs depend on it.

Also writes data/samples_<case>_targeted.csv. Inside a gate an ON transistor has drain and source at
the same rail, |Vds| below 1 mV in 90 % of the gate simulations, and drawing Vd and Vs independently
almost never lands there. Models trained on the uniform set alone missed those transistors' gate and
source currents by 40 to 400 % (median), and the gate leakage by 66 %. The targeted set draws the
same VDD, temperature, width and process values, and only Vd and Vs differently:
- near equal (2/3 of rows): one terminal at an anchor, a rail in 70 % of rows and uniform otherwise,
  the other offset from it by 10^U(-9, -0.5) V, kept inside 0 to VDD
- at the rails (1/3): each terminal at 0 or VDD, moved inwards by 10^U(-9, -2) V

Also writes data/samples_<case>_gatebias.csv, the bias a transistor has inside a static gate. A static
gate draws no DC current through an ON transistor, so both its ends sit at the rail it passes (0 V for
NMOS, VDD for PMOS). An OFF transistor either blocks between the two rails, with its source side
possibly a stack node a little off its rail, or sits beside an ON transistor with both ends at one
rail. VDD, temperature and process are drawn as for the other sets, W log-uniform over the same range:
- ON: drain and source each at the passed rail, moved inwards by 10^U(-9, -3) V
- OFF across the rails (2/3): the source side at the device's own rail (0 V for NMOS, VDD for PMOS)
  moved inwards by 10^U(-9, -1) V, the drain side at the other rail moved inwards by 10^U(-9, -3) V
- OFF at one rail (1/3): both ends at the same rail, either one, moved inwards by 10^U(-9, -3) V
Drain and source swap in half the rows, since a netlist can name either end the drain.
"""

import sys

import numpy as np
import pandas as pd

from common import (CASES, DATA_DIR, PROCESS_PARAMETERS, PROCESS_SPREAD, SAMPLE_SETS, TEMP_RANGE, VDD_NOMINAL,
                    VDD_SPREAD, WIDTH_RANGE, nominal_process_values)

SEED = 463
# The targeted streams are seeded apart from the uniform ones, so the uniform set is unchanged.
TARGETED_STREAM_OFFSET = 100
NEAR_EQUAL_FRACTION = 2 / 3
RAIL_ANCHOR_FRACTION = 0.7
NEAR_EQUAL_OFFSET_DECADES = (-9.0, -0.5)
RAIL_OFFSET_DECADES = (-9.0, -2.0)
# The gate-bias streams, seeded apart from both earlier sets so those stay byte-identical.
GATEBIAS_STREAM_OFFSET = 200
GATEBIAS_RAIL_DECADES = (-9.0, -3.0)
GATEBIAS_STACK_DECADES = (-9.0, -1.0)
GATEBIAS_ACROSS_FRACTION = 2 / 3


def gatebias_terminals(case_name, vdd, generator):
    """Drain and source voltages of a transistor inside a static gate, see the module docstring."""
    case = CASES[case_name]
    count = len(vdd)

    def inwards_from(at_vdd, decades):
        offset = 10.0 ** generator.uniform(*decades, count)
        return np.where(at_vdd, vdd - offset, offset)

    own_rail_high = case["device"] == "pmos"
    if case_name.endswith("_on"):
        drain = inwards_from(np.full(count, own_rail_high), GATEBIAS_RAIL_DECADES)
        source = inwards_from(np.full(count, own_rail_high), GATEBIAS_RAIL_DECADES)
    else:
        across = generator.uniform(size=count) < GATEBIAS_ACROSS_FRACTION
        shared_rail_high = generator.uniform(size=count) < 0.5
        source = np.where(across, inwards_from(np.full(count, own_rail_high), GATEBIAS_STACK_DECADES),
                          inwards_from(shared_rail_high, GATEBIAS_RAIL_DECADES))
        drain = np.where(across, inwards_from(np.full(count, not own_rail_high), GATEBIAS_RAIL_DECADES),
                         inwards_from(shared_rail_high, GATEBIAS_RAIL_DECADES))
    swap = generator.uniform(size=count) < 0.5
    return np.where(swap, source, drain), np.where(swap, drain, source)


def targeted_terminals(vdd, generator):
    """Drain and source voltages near each other or at the rails, as the transistors of a gate see them."""
    count = len(vdd)
    near_equal = generator.uniform(size=count) < NEAR_EQUAL_FRACTION

    at_rail = generator.uniform(size=count) < RAIL_ANCHOR_FRACTION
    high_rail = generator.uniform(size=count) < 0.5
    anchor = np.where(at_rail, np.where(high_rail, vdd, 0.0), generator.uniform(0, 1, count) * vdd)
    offset = 10.0 ** generator.uniform(*NEAR_EQUAL_OFFSET_DECADES, count)
    other = anchor + np.where(generator.uniform(size=count) < 0.5, offset, -offset)
    # The offset is at most 0.32 V, so reflecting at a rail stays inside 0 to VDD.
    other = np.where((other < 0) | (other > vdd), 2 * anchor - other, other)
    anchor_is_drain = generator.uniform(size=count) < 0.5
    near_drain = np.where(anchor_is_drain, anchor, other)
    near_source = np.where(anchor_is_drain, other, anchor)

    def at_a_rail():
        inwards = 10.0 ** generator.uniform(*RAIL_OFFSET_DECADES, count)
        return np.where(generator.uniform(size=count) < 0.5, inwards, vdd - inwards)

    rail_drain, rail_source = at_a_rail(), at_a_rail()
    return np.where(near_equal, near_drain, rail_drain), np.where(near_equal, near_source, rail_source)


def draw_samples(case_name, sample_count, generator, nominal, set_name="uniform"):
    case = CASES[case_name]
    vdd = generator.uniform(VDD_NOMINAL * (1 - VDD_SPREAD), VDD_NOMINAL * (1 + VDD_SPREAD), sample_count)
    temp = generator.uniform(*TEMP_RANGE, sample_count)
    if set_name == "gatebias":
        width = np.exp(generator.uniform(*np.log(WIDTH_RANGE), sample_count))
    else:
        width = generator.uniform(*WIDTH_RANGE, sample_count)
    columns = {
        "vdd": vdd,
        "temp": temp,
        "w": width,
        "vd": generator.uniform(0, 1, sample_count) * vdd,
        "vs": generator.uniform(0, 1, sample_count) * vdd,
        "vg": case["gate"] * vdd,
        "vb": case["body"] * vdd,
    }
    for parameter in PROCESS_PARAMETERS:
        factor = generator.uniform(1 - PROCESS_SPREAD, 1 + PROCESS_SPREAD, sample_count)
        columns[parameter] = nominal[case["device"]][parameter] * factor
    if set_name == "targeted":
        columns["vd"], columns["vs"] = targeted_terminals(vdd, generator)
    elif set_name == "gatebias":
        columns["vd"], columns["vs"] = gatebias_terminals(case_name, vdd, generator)
    return pd.DataFrame(columns)


def main():
    counts = {"uniform": int(sys.argv[1]) if len(sys.argv) > 1 else 10_000,
              "targeted": int(sys.argv[2]) if len(sys.argv) > 2 else 5_000,
              "gatebias": int(sys.argv[3]) if len(sys.argv) > 3 else 10_000}
    stream_offsets = {"uniform": 0, "targeted": TARGETED_STREAM_OFFSET, "gatebias": GATEBIAS_STREAM_OFFSET}
    nominal = nominal_process_values()
    DATA_DIR.mkdir(exist_ok=True)
    for set_name, suffix in SAMPLE_SETS.items():
        for case_index, case_name in enumerate(CASES):
            # One stream per case, so changing the sample count of one case leaves the others unchanged.
            generator = np.random.default_rng([SEED, stream_offsets[set_name] + case_index])
            samples = draw_samples(case_name, counts[set_name], generator, nominal, set_name)
            path = DATA_DIR / f"samples_{case_name}{suffix}.csv"
            samples.to_csv(path, index=False)
            print(f"{path.relative_to(DATA_DIR.parent)}: {len(samples)} rows")


if __name__ == "__main__":
    main()
