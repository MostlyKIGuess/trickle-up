"""Static CMOS gates, the 3-bit carry-lookahead adder built from them, and the leakage bookkeeping.

One definition of each gate's transistors feeds the SPICE decks, the oracle control and the ML
composition, so the three cannot disagree about which terminal sits on which node.

Leakage of a gate or circuit is the total current delivered by every node held at VDD: the supply
rail plus every primary input at logic 1. That equals static power / VDD. A transistor terminal
contributes its current (positive into the terminal) when its node is one of those nodes.
"""

import itertools
import re
from dataclasses import dataclass, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (L_MIN, PROCESS_PARAMETERS, SPICE_OPTIONS, TEMP_NOMINAL, VDD_NOMINAL, diffusion_geometry,
                    nominal_process_values)
from spice import model_cards

SUPPLY = "vdd"
GROUND = "0"
TERMINALS = ("drain", "gate", "source", "body")
CURRENT_COLUMNS = ["id", "ig", "is", "ib"]
CURRENT_OF_TERMINAL = dict(zip(TERMINALS, CURRENT_COLUMNS))
# Columns of the transistor models' input frame, in the order of the Part A datasets.
MODEL_INPUT_COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROCESS_PARAMETERS]


@dataclass(frozen=True)
class Transistor:
    name: str
    device: str
    drain: str
    gate: str
    source: str
    body: str
    width: float

    def node(self, terminal: str) -> str:
        return getattr(self, terminal)


@dataclass(frozen=True)
class Gate:
    inputs: tuple[str, ...]
    output: str
    internal: tuple[str, ...]
    transistors: tuple[Transistor, ...]


def _nmos(name: str, drain: str, gate: str, source: str, width_in_lmin: int) -> Transistor:
    return Transistor(name, "nmos", drain, gate, source, GROUND, width_in_lmin * L_MIN)


def _pmos(name: str, drain: str, gate: str, source: str, width_in_lmin: int) -> Transistor:
    return Transistor(name, "pmos", drain, gate, source, SUPPLY, width_in_lmin * L_MIN)


def _relabel(transistors: tuple[Transistor, ...], node_map: dict[str, str], name_suffix: str = "") -> tuple[Transistor, ...]:
    """Rename nodes found in node_map and append name_suffix to every transistor name."""
    return tuple(
        replace(transistor, name=transistor.name + name_suffix,
                **{terminal: node_map.get(transistor.node(terminal), transistor.node(terminal)) for terminal in TERMINALS})
        for transistor in transistors)


# Widths in units of Lmin = 22 nm. Series devices are doubled (NAND2 NMOS) or quadrupled (NOR2 PMOS)
# relative to the INV so the worst-case drive matches.
INVERTER = (_nmos("mn", "out", "a", GROUND, 1), _pmos("mp", "out", "a", SUPPLY, 2))
# Stack order follows the TA's AND2 example: B at the output, A at ground.
NAND2 = (
    _nmos("mnb", "out", "b", "x", 2),
    _nmos("mna", "x", "a", GROUND, 2),
    _pmos("mpa", "out", "a", SUPPLY, 2),
    _pmos("mpb", "out", "b", SUPPLY, 2),
)
# A at the supply, B at the output.
NOR2 = (
    _pmos("mpa", "y", "a", SUPPLY, 4),
    _pmos("mpb", "out", "b", "y", 4),
    _nmos("mna", "out", "a", GROUND, 1),
    _nmos("mnb", "out", "b", GROUND, 1),
)

GATES = {
    "inv": Gate(("a",), "out", (), INVERTER),
    "nand2": Gate(("a", "b"), "out", ("x",), NAND2),
    "nor2": Gate(("a", "b"), "out", ("y",), NOR2),
    "and2": Gate(("a", "b"), "out", ("x", "inner"),
                 _relabel(NAND2, {"out": "inner"}) + _relabel(INVERTER, {"a": "inner"}, "i")),
    "or2": Gate(("a", "b"), "out", ("y", "inner"),
                _relabel(NOR2, {"out": "inner"}) + _relabel(INVERTER, {"a": "inner"}, "i")),
}
# Every internal node name used by any gate, for the voltage columns of the gate dataset.
INTERNAL_NODES = ("x", "y", "inner")

LOGIC = {
    "inv": lambda a: 1 - a,
    "nand2": lambda a, b: 1 - (a & b),
    "nor2": lambda a, b: 1 - (a | b),
    "and2": lambda a, b: a & b,
    "or2": lambda a, b: a | b,
}


def input_combinations(gate_name: str) -> list[str]:
    """Input combinations as strings with the first input first, e.g. "01" is a = 0, b = 1."""
    return ["".join(bits) for bits in itertools.product("01", repeat=len(GATES[gate_name].inputs))]


# The 18 gate and input cases of the brief.
GATE_CASES = [(gate_name, inputs) for gate_name in GATES for inputs in input_combinations(gate_name)]


def gate_nodes_held_high(gate_name: str, inputs: str) -> set[str]:
    gate = GATES[gate_name]
    return {SUPPLY} | {node for node, bit in zip(gate.inputs, inputs) if bit == "1"}


def terminal_weights(transistors: tuple[Transistor, ...], held_high: set[str]) -> pd.DataFrame:
    """1 for each terminal whose node is held at VDD, else 0. Index: transistor name, columns: id, ig, is, ib."""
    return pd.DataFrame(
        [[float(transistor.node(terminal) in held_high) for terminal in TERMINALS] for transistor in transistors],
        index=[transistor.name for transistor in transistors], columns=CURRENT_COLUMNS)


def corner_process(corner: dict) -> dict:
    return {device: {parameter: corner[f"{device}_{parameter}"] for parameter in PROCESS_PARAMETERS}
            for device in ("nmos", "pmos")}


def nominal_corner() -> dict:
    """VDD 1 V, 27 C and the model file's process values, in the columns of data/corners.csv."""
    nominal = nominal_process_values()
    corner = {"corner": -1, "vdd": VDD_NOMINAL, "temp": TEMP_NOMINAL}
    corner |= {f"{device}_{parameter}": nominal[device][parameter]
               for device in ("nmos", "pmos") for parameter in PROCESS_PARAMETERS}
    return corner


def without_gate_resistor(model_text: str) -> str:
    """Set rgatemod = 0 in both model cards. Used by the gate and adder decks, not by Part A.

    With rgatemod = 1 the gate resistor is rshg * (W / 3) / L = 0.13 ohm (rshg = 0.4, xgw = 0), about
    7.5 S. A gate driven by another gate's output near VDD then carries about 1.1 V * 2.2e-16 * 7.5 S
    = 2e-15 A of rounding noise, far above abstol = 1e-18 A. Over the 200 Part B corners
    (results/rgatemod_comparison.csv), 61 of 3600 gate cases, all AND2, do not converge with
    rgatemod = 1, and 2 more converge to a point whose source-based leakage is 3 and 14 times off the
    device-based one. The resistor carries no DC current of its own, and on the other 3537 the two
    leakages agree within 0.51 % (99 % of them within 5e-5). rgatemod = 0 converges in all 3600.
    """
    edited, count = re.subn(r"\brgatemod\s*=\s*1\b", "rgatemod= 0", model_text)
    if count != 2:
        raise ValueError(f"expected rgatemod = 1 in both model cards, found it {count} times")
    return edited


def circuit_model_cards(corner: dict) -> str:
    """Model cards of the gate and adder decks: the corner's process values, without the gate resistor."""
    return without_gate_resistor(model_cards(corner_process(corner)))


# ---- SPICE decks -------------------------------------------------------------------------------

def transistor_line(transistor: Transistor, diffusion: bool = True) -> str:
    """Device line. diffusion = False drops AD, AS, PD and PS, as in the TA netlist's device line."""
    geometry = f" {diffusion_geometry(transistor.width)}" if diffusion else ""
    return (f"{transistor.name} {transistor.drain} {transistor.gate} {transistor.source} {transistor.body} "
            f"{transistor.device} W={transistor.width:.6e} L={L_MIN:.6e}{geometry}")


def device_current_lines(transistor: Transistor) -> list[str]:
    """Control lines defining <name>_id, _ig, _ib and _is, positive into each terminal.

    They come from BSIM4 device quantities, not from source currents, because a gate or body held at
    VDD reads a floating-point floor near 1e-16 A through the internal gate and body resistors. The
    terminal sums follow the right-hand-side stamps of ngspice-45 b4ld.c. BSIM4 works in a frame where
    the PMOS voltages and currents are negated (hence the sign) and, when that frame's vds is negative,
    drain and source are swapped (mode -1). @m[id] is only the channel current of that frame, so the
    drain terminal also collects GIDL, the drain junction and the gate-to-drain tunnelling, and the
    channel current, impact ionisation current and channel tunnelling (igcs or igcd) belong to the
    terminal acting as drain:
      mode +1: id = cd + isub + igidl - ibd - igd - igcd
      mode -1: id = -cd + igidl - ibd - igd - igcs
    Gate and body sums are the same in both modes. The source current closes Kirchhoff's law.
    """
    name = transistor.name
    sign = "" if transistor.device == "nmos" else "-"

    def quantity(parameter: str) -> str:
        return f"@{name}[{parameter}]"

    return [
        f"let {name}_fwd = {quantity('vds')} ge 0",
        f"let {name}_id = {sign}({name}_fwd * ({quantity('id')} + {quantity('isub')} - {quantity('igcd')})"
        f" - (1 - {name}_fwd) * ({quantity('id')} + {quantity('igcs')})"
        f" + {quantity('igidl')} - {quantity('ibd')} - {quantity('igd')})",
        f"let {name}_ig = {sign}({quantity('igs')} + {quantity('igd')} + {quantity('igcs')} + {quantity('igcd')}"
        f" + {quantity('igb')})",
        f"let {name}_ib = {sign}({quantity('ibd')} + {quantity('ibs')} - {quantity('igidl')} - {quantity('igisl')}"
        f" - {quantity('isub')} - {quantity('igb')})",
        f"let {name}_is = -({name}_id + {name}_ig + {name}_ib)",
    ]


def _print_lines(names: list[str], per_line: int = 8) -> list[str]:
    return [f"print {' '.join(names[start:start + per_line])}" for start in range(0, len(names), per_line)]


def leakage_deck(title: str, transistors: list[Transistor], sources: dict[str, float], held_high: set[str],
                 temp: float, model_text: str, probe_nodes: list[str],
                 print_transistor_currents: bool, diffusion: bool = True) -> tuple[str, list[str]]:
    """Operating-point deck that prints the device-based and source-based leakage and the probe voltages.

    sources maps each driven node (the supply and the inputs) to its voltage. Returns the deck text and
    the names it prints.
    """
    source_lines = [f"v_{node} {node} {GROUND} {voltage:.9f}" for node, voltage in sources.items()]
    contributing = [transistor for transistor in transistors
                    if any(transistor.node(terminal) in held_high for terminal in TERMINALS)]
    current_owners = transistors if print_transistor_currents else contributing
    current_lines = [line for transistor in current_owners for line in device_current_lines(transistor)]
    # Summed node by node, independently of the per-transistor weights used by the composition code.
    leakage_lines = ["let leak_device = 0"]
    for node in sorted(held_high):
        for transistor in transistors:
            for terminal in TERMINALS:
                if transistor.node(terminal) == node:
                    leakage_lines.append(
                        f"let leak_device = leak_device + {transistor.name}_{CURRENT_OF_TERMINAL[terminal]}")
    # i(v) is the current from the + node through the source, so the current a source delivers is -i(v).
    source_terms = " ".join(f"- i(v_{node})" for node in sorted(held_high))
    leakage_lines.append(f"let leak_source = 0 {source_terms}")
    probe_lines = [f"let v_{node} = v({node})" for node in probe_nodes]

    printed = ["leak_device", "leak_source", *[f"v_{node}" for node in probe_nodes]]
    if print_transistor_currents:
        printed += [f"{transistor.name}_{current}" for transistor in transistors for current in CURRENT_COLUMNS]
    deck = "\n".join([
        f"* {title}",
        model_text,
        SPICE_OPTIONS,
        f".temp {temp:.6f}",
        "",
        *[transistor_line(transistor, diffusion) for transistor in transistors],
        *source_lines,
        "",
        ".control",
        # Twelve printed digits, so the composition controls can reach reltol = 1e-6.
        "set numdgt=12",
        # ngspice is built with OpenMP and defaults to 2 device-evaluation threads per process. With
        # eight processes in a pool they oversubscribe the cores: one adder deck took 3.8 s instead of
        # 0.68 s, with the same printed leakage.
        "set num_threads=1",
        "op",
        *current_lines,
        *leakage_lines,
        *probe_lines,
        *_print_lines(printed),
        ".endc",
        ".end",
        "",
    ])
    return deck, printed


def gate_deck(gate_name: str, inputs: str, corner: dict, model_text: str,
              diffusion: bool = True) -> tuple[str, list[str]]:
    gate = GATES[gate_name]
    vdd = corner["vdd"]
    sources = {SUPPLY: vdd} | {node: vdd * int(bit) for node, bit in zip(gate.inputs, inputs)}
    return leakage_deck(f"{gate_name} inputs {inputs}", list(gate.transistors), sources,
                        gate_nodes_held_high(gate_name, inputs), corner["temp"], model_text,
                        [gate.output, *gate.internal], print_transistor_currents=True, diffusion=diffusion)


# ---- 3-bit carry-lookahead adder ---------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    name: str
    gate: str
    inputs: tuple[str, ...]
    output: str


def _xor(output: str, first: str, second: str) -> list[Instance]:
    """XOR from four NAND2: n1 = NAND(a, b), n2 = NAND(a, n1), n3 = NAND(b, n1), out = NAND(n2, n3)."""
    return [
        Instance(f"{output}x1", "nand2", (first, second), f"{output}x1"),
        Instance(f"{output}x2", "nand2", (first, f"{output}x1"), f"{output}x2"),
        Instance(f"{output}x3", "nand2", (second, f"{output}x1"), f"{output}x3"),
        Instance(f"{output}x4", "nand2", (f"{output}x2", f"{output}x3"), output),
    ]


def _single(gate_name: str, output: str, first: str, second: str) -> list[Instance]:
    return [Instance(output, gate_name, (first, second), output)]


CLA_INPUTS = ("a0", "a1", "a2", "b0", "b1", "b2", "cin")
CLA_OUTPUTS = ("s0", "s1", "s2", "c3")
# In evaluation order. P_i = a_i XOR b_i, G_i = a_i AND b_i, each carry is a sum of products of
# P, G and cin (C0 = cin), and S_i = P_i XOR C_i. The partial products p10, p21 and p210 are shared.
CLA_INSTANCES = [
    *_xor("p0", "a0", "b0"), *_xor("p1", "a1", "b1"), *_xor("p2", "a2", "b2"),
    *_single("and2", "g0", "a0", "b0"), *_single("and2", "g1", "a1", "b1"), *_single("and2", "g2", "a2", "b2"),
    # C1 = G0 + P0 C0
    *_single("and2", "t10", "p0", "cin"), *_single("or2", "c1", "g0", "t10"),
    # C2 = G1 + P1 G0 + P1 P0 C0
    *_single("and2", "t20", "p1", "g0"), *_single("and2", "p10", "p1", "p0"), *_single("and2", "t21", "p10", "cin"),
    *_single("or2", "o20", "g1", "t20"), *_single("or2", "c2", "o20", "t21"),
    # C3 = G2 + P2 G1 + P2 P1 G0 + P2 P1 P0 C0
    *_single("and2", "t30", "p2", "g1"), *_single("and2", "p21", "p2", "p1"), *_single("and2", "t31", "p21", "g0"),
    *_single("and2", "p210", "p21", "p0"), *_single("and2", "t32", "p210", "cin"),
    *_single("or2", "o30", "g2", "t30"), *_single("or2", "o31", "t31", "t32"), *_single("or2", "c3", "o30", "o31"),
    *_xor("s0", "p0", "cin"), *_xor("s1", "p1", "c1"), *_xor("s2", "p2", "c2"),
]
CLA_NETS = [instance.output for instance in CLA_INSTANCES]
VECTOR_COUNT = 2 ** len(CLA_INPUTS)


def instantiate(instance: Instance) -> tuple[Transistor, ...]:
    """The gate's transistors with ports mapped to the instance's nets and private internal nodes."""
    gate = GATES[instance.gate]
    node_map = dict(zip(gate.inputs, instance.inputs)) | {gate.output: instance.output}
    node_map |= {node: f"{node}_{instance.name}" for node in gate.internal}
    return _relabel(gate.transistors, node_map, f"_{instance.name}")


CLA_TRANSISTORS = [transistor for instance in CLA_INSTANCES for transistor in instantiate(instance)]


def vector_operands(vector: int) -> tuple[int, int, int]:
    """Vector index to (a, b, cin): bits 0-2 are a, bits 3-5 are b, bit 6 is cin."""
    return vector & 7, (vector >> 3) & 7, (vector >> 6) & 1


def vector_input_levels(vector: int) -> dict[str, int]:
    a_value, b_value, carry_in = vector_operands(vector)
    levels = {f"a{bit}": (a_value >> bit) & 1 for bit in range(3)}
    levels |= {f"b{bit}": (b_value >> bit) & 1 for bit in range(3)}
    levels["cin"] = carry_in
    return levels


def cla_logic(vector: int) -> dict[str, int]:
    """Logic level of every input and gate output net for one input vector."""
    levels = vector_input_levels(vector)
    for instance in CLA_INSTANCES:
        levels[instance.output] = LOGIC[instance.gate](*[levels[net] for net in instance.inputs])
    return levels


def cla_gate_cases(vector: int) -> list[tuple[str, str]]:
    """(gate, input combination) of every CLA instance for one input vector."""
    levels = cla_logic(vector)
    return [(instance.gate, "".join(str(levels[net]) for net in instance.inputs)) for instance in CLA_INSTANCES]


def cla_deck(vector: int, corner: dict, model_text: str) -> tuple[str, list[str]]:
    vdd = corner["vdd"]
    levels = vector_input_levels(vector)
    sources = {SUPPLY: vdd} | {net: vdd * levels[net] for net in CLA_INPUTS}
    held_high = {SUPPLY} | {net for net in CLA_INPUTS if levels[net] == 1}
    return leakage_deck(f"3-bit carry-lookahead adder, vector {vector}", CLA_TRANSISTORS, sources, held_high,
                        corner["temp"], model_text, CLA_NETS, print_transistor_currents=False)


# ---- Comparison ---------------------------------------------------------------------------------

def comparison_metrics(spice: pd.Series, predicted: pd.Series) -> dict:
    """Relative error (predicted - SPICE) / SPICE. Bias is its mean, the error columns use its magnitude."""
    spice_values = spice.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    if len(spice_values) == 0 or np.isnan(spice_values).any() or np.isnan(predicted_values).any():
        raise ValueError(f"comparison needs non-empty values without NaN, got {len(spice_values)} rows")
    relative = (predicted_values - spice_values) / spice_values
    return {
        "simulations": len(spice_values),
        "spice_median_A": np.median(spice_values),
        "predicted_median_A": np.median(predicted_values),
        "bias_percent": 100 * relative.mean(),
        "mean_error_percent": 100 * np.abs(relative).mean(),
        "median_error_percent": 100 * np.median(np.abs(relative)),
        "p95_error_percent": 100 * np.percentile(np.abs(relative), 95),
        "max_error_percent": 100 * np.abs(relative).max(),
    }


GATE_COLORS = {"inv": "#2a78d6", "nand2": "#eb6834", "nor2": "#1baf7a", "and2": "#eda100", "or2": "#e87ba4"}


def parity_plot(groups: dict[str, tuple[np.ndarray, np.ndarray]], colors: dict[str, str], x_label: str,
                y_label: str, path_stem) -> None:
    """Log-log scatter of predicted against SPICE leakage, one colour per group, with the y = x line."""
    figure, axis = plt.subplots(figsize=(4, 4))
    all_values = np.concatenate([np.concatenate(pair) for pair in groups.values()])
    limits = (all_values.min() / 1.5, all_values.max() * 1.5)
    axis.plot(limits, limits, color="#888888", linewidth=1, zorder=1)
    for label, (spice_values, predicted_values) in groups.items():
        axis.scatter(spice_values, predicted_values, s=4, alpha=0.5, color=colors[label], label=label,
                     rasterized=True, zorder=2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if len(groups) > 1:
        axis.legend(frameon=False, markerscale=3)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(f"{path_stem}.{suffix}", dpi=200)
    plt.close(figure)
