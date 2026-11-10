"""Simulate every sample of data/samples_<case><suffix>.csv with ngspice and write data/fet_<case><suffix>.csv.

Usage: python3 scripts/02_sim_fet.py [max_samples_per_case | all] [workers] [uniform | targeted | gatebias]

The suffix is "", "_targeted" or "_gatebias", see SAMPLE_SETS in common.py.

Without a set name both sample sets are simulated.

Each sample is one deck: its own copy of the model card with the drawn process values, its own
temperature, and a voltage source on every terminal. Currents are positive into the transistor
terminal, so the four of them sum to zero (the Kirchhoff check of 03_check_data.py).

The gate and body currents are the device's own sums of its components, not the gate and body
source currents. The model has internal gate (rgatemod = 1) and body (rbodymod = 1) resistor nodes,
and the source current of a gate or body held at VDD carries a floating-point floor near 1e-16 A
(a PMOS OFF sample read -8.9e-16 A at the gate source and -8.1e-22 A in the device). Where the
currents are large the two agree: 3.4456e-10 A both ways for an NMOS ON gate, -1.06820e-11 A at the
source and -1.068195e-11 A in the device for an NMOS ON body. Drain and source have no internal
resistors (rdsmod = 0), so their source currents are exact and the Kirchhoff check stays independent.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from common import (CASES, DATA_COLUMNS, DATA_DIR, L_MIN, PROCESS_PARAMETERS, SAMPLE_SETS, SPICE_DIR, SPICE_OPTIONS,
                    diffusion_geometry)
from spice import model_cards, ngspice_version, run_deck

CURRENT_NAMES = ["cur_d", "cur_g", "cur_s", "cur_b"]


def fet_deck(sample, device):
    process = {parameter: sample[parameter] for parameter in PROCESS_PARAMETERS}
    # BSIM4 reports the PMOS components with the opposite sign to the terminal current.
    sign = "" if device == "nmos" else "-"
    return f"""* single {device}, one terminal source per terminal
{model_cards({device: process})}
{SPICE_OPTIONS}
.temp {sample['temp']:.6f}

M1 drain gate source body {device} W={sample['w']:.6e} L={L_MIN:.6e} {diffusion_geometry(sample['w'])}
Vd drain 0 {sample['vd']:.12e}
Vg gate 0 {sample['vg']:.12e}
Vs source 0 {sample['vs']:.12e}
Vb body 0 {sample['vb']:.12e}

.control
set numdgt=12
* The pool runs one deck per worker, so ngspice's own OpenMP threads only compete with it.
set num_threads=1
op
* i(V) is the current from the + node through the source, which is the current out of the
* transistor terminal. The sign is flipped so each current is positive into the terminal.
let cur_d = -i(vd)
* Gate: tunnelling to source and drain overlaps, to the channel, and to the body.
let cur_g = {sign}(@m1[igs] + @m1[igd] + @m1[igcs] + @m1[igcd] + @m1[igb])
let cur_s = -i(vs)
* Body: junction currents into the body, minus GIDL, GISL, impact ionisation and gate-to-body current,
* which all leave through the body terminal.
let cur_b = {sign}(@m1[ibd] + @m1[ibs] - @m1[igidl] - @m1[igisl] - @m1[isub] - @m1[igb])
print cur_d cur_g cur_s cur_b
.endc
.end
"""


def simulate(job):
    sample, device = job
    return run_deck(fet_deck(sample, device), CURRENT_NAMES)


def write_example_netlists():
    """spice/fet_<case>.cir, the deck of the first uniform sample of each case, for the deliverables."""
    SPICE_DIR.mkdir(exist_ok=True)
    for case_name, case in CASES.items():
        sample = pd.read_csv(DATA_DIR / f"samples_{case_name}.csv").iloc[0].to_dict()
        (SPICE_DIR / f"fet_{case_name}.cir").write_text(fet_deck(sample, case["device"]))


def main():
    write_example_netlists()
    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    set_names = [sys.argv[3]] if len(sys.argv) > 3 else list(SAMPLE_SETS)
    (DATA_DIR / "ngspice_version.txt").write_text(ngspice_version() + "\n")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for set_name in set_names:
            suffix = SAMPLE_SETS[set_name]
            for case_name, case in CASES.items():
                samples = pd.read_csv(DATA_DIR / f"samples_{case_name}{suffix}.csv")
                if max_samples is not None:
                    samples = samples.head(max_samples)
                jobs = [(row, case["device"]) for row in samples.to_dict("records")]
                currents = pd.DataFrame(list(pool.map(simulate, jobs, chunksize=32)))
                currents.columns = ["id", "ig", "is", "ib"]
                dataset = pd.concat([samples.reset_index(drop=True), currents], axis=1)[DATA_COLUMNS]
                path = DATA_DIR / f"fet_{case_name}{suffix}.csv"
                dataset.to_csv(path, index=False, float_format="%.10e")
                print(f"{path.relative_to(DATA_DIR.parent)}: {len(dataset)} rows")


if __name__ == "__main__":
    main()
