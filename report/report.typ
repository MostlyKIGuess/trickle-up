// Compile from the repository root: typst compile --root . report/report.typ report/report.pdf
// Every table reads its numbers from results/ or report/analysis/ (report/error_analysis.py and
// report/summarize_models.py), so the tables and the CSV files cannot disagree.

#let accent = rgb("#1c4e8a")
#let accent-light = rgb("#eaf1fb")
// Table cells: accent-light is the result to read, warn-light a large value the text explains.
#let warn-light = rgb("#f8e9e2")
// In a table head: lower is better, higher is better.
#let good-down = $arrow.b$
#let good-up = $arrow.t$

#set document(title: "ML-driven PVT variation aware leakage estimation")
#set page(paper: "a4", margin: (x: 2.1cm, top: 2.2cm, bottom: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
// sticky keeps each heading on the same page as the text after it.
#show heading.where(level: 1): it => block(sticky: true, above: 1.4em, below: 0.8em)[
  #text(size: 14pt, weight: "bold", fill: accent)[#if it.numbering != none [#counter(heading).display() ]#it.body]
]
#show heading.where(level: 2): it => block(sticky: true, above: 1.1em, below: 0.6em)[
  #text(size: 11.5pt, weight: "bold", fill: accent)[#counter(heading).display() #it.body]
]
#show raw: set text(font: "DejaVu Sans Mono", size: 9pt)
#show figure.caption: set text(size: 9.5pt)
// Image figures float to the top or bottom of a page, so a large one does not leave half a page empty.
#show figure.where(kind: image): set figure(placement: auto)
#show figure.where(kind: table): set figure(placement: auto)
#show figure.where(kind: table): set figure.caption(position: top)
#show table: set par(justify: false)
#set table(stroke: 0.5pt + luma(180), inset: 5pt)
#show table.cell.where(y: 0): set text(weight: "bold")

// ---- Numbers from the CSV files ---------------------------------------------------------------

#let load(path) = csv(path, row-type: dictionary)

// value rounded to a fixed number of decimals, with trailing zeros kept.
#let fixed(value, digits) = {
  let rounded = calc.round(float(value), digits: digits)
  if digits == 0 { return str(int(rounded)) }
  let parts = str(calc.abs(rounded)).split(".")
  let fraction = if parts.len() > 1 { parts.at(1) } else { "" }
  let sign = if rounded < 0 { "−" } else { "" }
  sign + parts.at(0) + "." + fraction + "0" * (digits - fraction.len())
}

// value as m x 10^e.
#let sci(value, digits: 2) = {
  let number = float(value)
  if number == 0 { return [0] }
  let exponent = calc.floor(calc.log(calc.abs(number)))
  let mantissa = number / calc.pow(10.0, exponent)
  if calc.abs(calc.round(mantissa, digits: digits)) >= 10 {
    exponent += 1
    mantissa = mantissa / 10
  }
  $#fixed(mantissa, digits) times 10^(#str(exponent))$
}

// A percentage that may run to thousands, with fewer decimals as it grows.
#let pct-wide(percent) = {
  let number = float(percent)
  let size = calc.abs(number)
  if size >= 1000 { sci(number, digits: 1) } else if size >= 100 { fixed(number, 0) } else if size >= 10 { fixed(number, 1) } else if size >= 0.1 or size == 0 { fixed(number, 2) } else if size >= 0.001 { fixed(number, 4) } else { sci(number, digits: 1) }
}
// A fraction shown as a percentage.
#let pct(fraction, digits: 2) = fixed(100 * float(fraction), digits)
#let pct-of(fraction) = pct-wide(100 * float(fraction))

#let case-names = (nmos_on: "NMOS ON", nmos_off: "NMOS OFF", pmos_on: "PMOS ON", pmos_off: "PMOS OFF")
#let model-names = (null_median: "Median only", poly_ridge: "Poly ridge", extra_trees: "Extra trees",
  lightgbm: "LightGBM", mlp: "MLP")
#let gate-name(name) = upper(name)
#let small-table(body) = block(text(size: 8.5pt, body))

#let gate-ml = load("/results/gate_results_ml.csv")
#let gate-terminal = load("/results/gate_results_ml_terminal.csv")
#let gate-oracle = load("/results/gate_results_oracle.csv")
#let gate-oracle-terminal = load("/results/gate_results_oracle_terminal.csv")
#let cla-ml = load("/results/cla_results_ml.csv").at(0)
#let cla-terminal = load("/results/cla_results_ml_terminal.csv").at(0)
#let cla-oracle = load("/results/cla_results_oracle.csv").at(0)
#let cla-control = load("/results/cla_composition_control.csv").at(0)
#let gate-all = gate-ml.at(-1)
#let blind = load("/results/stage4/stage4_summary.csv")
#let blind-row(level, method) = blind.filter(row => row.level == level and row.method == method).at(0)
#let blind-gate = blind-row("gate", "frozen")
#let blind-adder = blind-row("adder", "frozen")
#let noise = load("/results/arms/noise_floor.csv").filter(row => row.at("") == "gatebias_score").at(0)
#let arms = load("/results/arms/arm_report.csv")
#let arm(label) = arms.find(row => row.label == label)
#let timing-rows = load("/results/timing.csv")
#let timing(name) = float(timing-rows.find(row => row.name == name).value)
// The speedup to the nearest ten.
#let speedup = fixed(10 * calc.round(timing("speedup") / 10), 0)

// ---- Title ------------------------------------------------------------------------------------

#align(center)[
  #text(size: 17pt, weight: "bold", fill: accent)[ML-driven PVT variation aware leakage estimation]
  #v(0.2em)
  Aditya Peketi, Krish Pandya, Sudhanva Joshi and Vignesh Vembar (team Dvd Player)
  #v(0.1em)
  #text(size: 9.5pt, fill: luma(90))[Digital VLSI Design (ECE463) course project. PTM 22nm LP, 3-bit carry-lookahead adder]
  #v(0.1em)
  #text(size: 9.5pt)[Code and data: #link("https://github.com/MostlyKIGuess/trickle-up")[github.com/MostlyKIGuess/trickle-up]]
]

#v(0.6em)
#block(fill: accent-light, inset: 10pt, radius: 3pt, width: 100%)[
  *Summary.* We predict the leakage current of CMOS circuits under process, voltage and temperature
  (PVT) variation from machine-learning models of single transistors. At each corner only the 18 gate
  cases are simulated, for their node voltages, and the circuit built from them is not.
  Over five gates and 200 random corners the predicted gate leakage is within
  #fixed(gate-all.mean_error_percent, 2) % of SPICE on average, and a 3-bit carry-lookahead adder
  built from those gates is within #fixed(cla-ml.mean_error_percent, 2) %. On 400 more corners, drawn
  only after the models were final, the errors are #fixed(blind-gate.mean_error_percent, 2) % and
  #fixed(blind-adder.mean_error_percent, 2) %. One leakage of the adder takes
  #fixed(timing("spice_seconds_per_adder_leakage"), 2) s in SPICE and
  #fixed(1000 * timing("method_seconds_per_adder_leakage"), 1) ms with our method, about #speedup times less.
]
= Introduction

The leakage of a circuit changes with supply voltage, temperature and process values. In our
simulations the subthreshold current of a single transistor grows by 2.6 decades between -40 and
125 °C, and a change of vth0 alone across its ±10 % range moves it by another 1.5 to 1.7 decades
(@fig-trends).
So the standby power of a circuit has to be known at many corners, where a corner is one combination
of the three, and for every input vector. SPICE gives that number exactly, but it needs one
simulation of the whole circuit for each corner and vector.

This project replaces those circuit simulations with machine learning. A model learns the four
terminal currents of a single transistor from its voltages, its width and the corner. The leakage of
a gate is then calculated from its transistors, and the leakage of a circuit from its gates
(@fig-composition). Only the transistor model is learned. The work has three parts (@tab-parts), and
we aim for a mean error below 1 % against SPICE.

#figure(
  placement: none,
  small-table(table(
    columns: (auto, 1.3fr, 0.8fr, 1.4fr),
    align: (left, left, left, left),
    table.header([Part], [Input], [Output], [Where SPICE is used]),
    [A, transistor], [One transistor of one case (NMOS or PMOS, ON or OFF) with its VDD, temperature,
      width, four terminal voltages and five process values],
      [Its four terminal currents Id, Ig, Is and Ib],
      [To make the samples the models are trained and tested on. A trained model runs without it.],
    [B, gate], [A gate, its input combination and a corner], [The leakage of the gate],
      [For the node voltages inside the gate, which the transistor models take as input, and for the
      reference leakage we compare with.],
    [C, adder], [An input vector of the adder and a corner], [The leakage of the adder],
      [Only for the reference leakage. The prediction is the sum of the predicted leakages of the 42
      gates from Part B at that corner.],
  )),
  caption: [The three parts of the work.],
) <tab-parts>

We did the work in this order.

+ *Data.* We drew 10,000 uniform random samples for each of the four transistor cases (NMOS and
  PMOS, each ON and OFF) and simulated every sample in SPICE (@sec-data).
+ *Baseline.* We compared four model types. An MLP that predicts log#sub[10]|I| has the lowest error
  in all four cases (@sec-models).
+ *The baseline misses the target.* On the gates its mean error is 2.07 %, twice the 1 % we aim for.
+ *Two causes.* An ON transistor in a gate has its drain and source at the same rail, with a median
  of 0.84 µV between them, and uniform samples almost never put two voltages that close. The drain
  and source currents of such a transistor also change sign near zero volts, and a model does not
  predict them well.
+ *Two fixes.* We add a targeted set of training samples with the drain and source close together
  (@sec-gap). We take the drain and source currents of the ON transistors from Kirchhoff's current
  law at the nodes of the gate and not from a model (@sec-compose).
+ *A better transistor model.* We tested five changes to the baseline on a separate sample set, so
  that the gates of Part B were never used to choose a model, and kept three (@sec-selection).
+ *A test on new corners.* We ran the final method once on 400 corners drawn only after the models
  were final (@sec-blind).

The mean error is below 1 % for the gates and for the adder, and on the new corners it is within
0.03 points of the corners we developed on (@tab-headline).

#figure(
  placement: none,
  table(
    columns: 3,
    align: (left, right, right),
    table.header([Mean error (%)~#good-down], [200 corners of Part B], [400 held-out corners]),
    [Five gates, 18 gate cases], fixed(gate-all.mean_error_percent, 2), fixed(blind-gate.mean_error_percent, 2),
    [3-bit carry-lookahead adder], fixed(cla-ml.mean_error_percent, 2), fixed(blind-adder.mean_error_percent, 2),
  ),
  caption: [Mean error of the predicted leakage against SPICE.],
) <tab-headline>

Four findings come out of the work.

- Accuracy on uniform test samples does not carry over to the gates. Models trained on the uniform
  set alone are accurate on their own test samples and miss the gate current of the ON transistors
  of a gate by 41 % (NMOS) and 116 % (PMOS) (@sec-gap).
- The currents a model predicts poorly do not have to be learned. Kirchhoff's current law gives the
  drain and source currents of the ON transistors. Without it the final models are off by
  #fixed(gate-terminal.at(-1).median_error_percent, 1) % in the median (@sec-compose).
- The leakage of a gate in this technology is mostly gate tunnelling of its ON transistors, 97.5 %
  over all simulations. The accuracy of that one current decides the accuracy of the gate
  (@sec-errors).
- Gate leakages can be added up. The sum of the SPICE leakages of the 42 gates matches the SPICE
  adder to 0.003 % on average, so the error of the adder is the error of its gates (@sec-adder).

== Setup

We define the leakage of a gate as the total current delivered by every node held at VDD, which is
the supply and every input at logic 1. That equals the static power divided by VDD.

The transistors are PTM 22nm LP (BSIM4), with a nominal VDD of 1 V, simulated with ngspice-45. The
test circuit is a 3-bit carry-lookahead adder of 42 gates and 204 transistors (@tab-setup).

#figure(
  placement: auto,
  image("/figures/architecture/composition.pdf", width: 94%),
  caption: [How a leakage is calculated. On the left is the adder with its inputs and outputs. Each symbol is one of its 42 gates (24 NAND2, 12 AND2 and 6 OR2), and each XOR is built from four NAND2. In the middle, one of those gates is enlarged, a NAND2 with a = 1 and b = 0. Each transistor has four small squares, one for each terminal current. A yellow square is a current predicted by a transistor model, and a blue square is a current we calculate from Kirchhoff's current law. The squares with a red outline are on a node held at VDD, and their sum is the gate leakage. On the right, the leakage of the adder is the sum of the leakages of its 42 gates.],
) <fig-composition>

= Data generation <sec-data>

== The uniform set <sec-sampling>

For each of the four cases we draw every input of a sample independently (@tab-uniform). This is the uniform set, 10,000 samples per case.

#figure(
  placement: none,
  small-table(table(
    columns: (auto, auto),
    align: (left, left),
    table.header([Input], [Drawn as]),
    [VDD], [$cal(U)(0.9, 1.1)$ V],
    [Vd, Vs], [$cal(U)(0, "VDD")$],
    [W], [$cal(U)(22, 176)$ nm, which is 1 to 8 Lmin],
    [Temperature], [$cal(U)(-40, 125)$ °C],
    [vth0, toxe, ndep, u0, xj], [$cal(U)(0.9, 1.1)$ times the value in the model file, each on its own,
      separately for NMOS and PMOS],
    [Vg], [VDD for NMOS ON and PMOS OFF, 0 for NMOS OFF and PMOS ON],
    [Vb], [0 for NMOS, VDD for PMOS],
  )),
  caption: [How a sample of the uniform set is drawn. $cal(U)(a, b)$ is a uniform draw between a and b.],
) <tab-uniform>

== Ground truth from SPICE <sec-netlists>

Given a sample, we get its four terminal currents from one ngspice simulation. The netlist has the
transistor, one voltage source per terminal, the sample's temperature, and a copy of the transistor
model with the sample's five process values. We run an operating-point analysis (`op`) and read the
four currents, counted as positive into the terminal. Id and Is are the currents of their voltage
sources. Ig and Ib are the gate and body currents that ngspice reports for the transistor itself.
Each transistor has drain and source areas `AD = AS = W * Ldiff` and perimeters
`PD = PS = 2 * (W + Ldiff)`, with `Ldiff = 44n`, which is 2 Lmin.

== Targeted and gate-bias sets <sec-gap>

Models trained on the uniform set alone are accurate on samples like the ones they were trained on
and fail on the ON transistors of a gate. For those transistors the predicted gate current is off by
41 % (NMOS) and 116 % (PMOS) in the median (@tab-transistor-errors).

The reason is that the uniform set does not cover the voltages a transistor has in a gate. @fig-vds
shows this through the voltage between drain and source, |Vds|. Each curve gives the share of
transistors whose |Vds| is below the value on the horizontal axis, which is logarithmic. A curve
that rises early has many transistors at small |Vds|. The left panel is for ON transistors and the
right panel for OFF transistors.

The black curve is the reference. In Part B we simulate the five gates in SPICE, for every input
combination and at 200 corners (@sec-reference), and each simulation gives the terminal voltages of
every transistor of the gate. The 18 gate cases have 84 transistors together, 42 ON and 42 OFF, so
the 200 corners give 84 × 200 = 16,800 transistors. An ON transistor in a static gate has its drain
and source at the same rail, so in the left panel the black curve rises between 0.1 and 10 µV and
reaches 90.5 % at 1 mV (the dot), with a median of 0.84 µV. The blue curve of the uniform set stays
near zero up to 10 mV, and only 0.2 % of its ON samples are below 1 mV. The uniform set therefore
has almost no samples where the ON transistors of a gate are. In the right panel only 19 % of the
OFF transistors are below 1 mV. The rest hold off a large part of the supply, where the uniform set
does have samples.

#figure(
  placement: auto,
  image("/report/analysis/sample_vds.pdf", width: 100%),
  caption: [Uniform samples do not cover the voltages an ON transistor has in a gate. Each curve is the cumulative share of |Vds|, for the transistors of the Part B gates and for the three sample sets. The dot marks 1 mV.],
) <fig-vds>

We therefore add two more sample sets (@tab-sets). Both draw VDD, temperature and process as the
uniform set does. They differ in how Vd and Vs are drawn.

*Targeted set, for training.* This set adds the samples the uniform set lacks, with Vd and Vs close
to each other. In two thirds of its samples one terminal is placed first and the other is set
between 1 nV and 0.3 V away from it. In the remaining third both terminals are placed near a rail,
0 or VDD. The models are trained on the uniform and targeted sets together. In @fig-vds it is the
grey curve, which covers every |Vds| from 1 nV upwards.

*Gate-bias set, for comparing models.* We also need to tell which of two models works better in
a gate, and the Part B gates must not be used to train or to pick a model. This set imitates the
voltages a transistor has in a gate. An ON NMOS has its drain and source both at 0 and an ON
PMOS has both at VDD, within 1 mV. An OFF transistor has either the full supply across it or both
ends at the same rail. No model is trained on this set. We use it only in @sec-selection to compare
models. In @fig-vds it is the green curve.

#figure(
  placement: auto,
  small-table(table(
    columns: (auto, auto, auto, 1fr),
    align: (left, right, left, left),
    table.header([Set], [Samples per case], [Used for], [How Vd and Vs are drawn]),
    [Uniform], [10,000], [Training, validation, test], [Each uniform in 0 to VDD.],
    [Targeted], [5,000], [Training, validation, test],
    [2/3 of the samples: one terminal at a rail in 70 % of these samples and uniform in 0 to VDD otherwise, the other away from it by $10^(cal(U)(-9, -0.5))$ V. 1/3 of the samples: each
      terminal at 0 or VDD, moved inwards by $10^(cal(U)(-9, -2))$ V.],
    [Gate-bias], [10,000], [Judging models],
    [ON: both at the rail the transistor passes, moved inwards by $10^(cal(U)(-9, -3))$ V. OFF across the
      rails (2/3): source side at the device's own rail moved inwards by $10^(cal(U)(-9, -1))$ V, drain
      side at the other rail moved inwards by $10^(cal(U)(-9, -3))$ V. OFF at one rail (1/3): both ends
      at the same rail, moved inwards by $10^(cal(U)(-9, -3))$ V. W is log-uniform.],
  )),
  caption: [The three sample sets and how each draws Vd and Vs. With the exponent $cal(U)(-9, -3)$ a voltage lies between 1 nV and 1 mV.],
) <tab-sets>

== Data check <sec-check>

The four terminal currents of a transistor must sum to zero, which gives a check on every sample.
For each sample we computed the residual |Id + Ig + Is + Ib| and divided it by the largest terminal
current of the same sample, so that ON and OFF cases can be compared. In @tab-kirchhoff the median
of this ratio is about 1e-11 in every case and sample set. The largest value (shaded in orange) is
in the targeted PMOS OFF set, in samples where Vd and Vs sit at the same rail and the drain and
source currents nearly cancel, so the ratio is large although the residual is small. As a current,
the largest residual of any sample is 9.6e-15 A.

#let kirchhoff = load("/results/kirchhoff.csv")
#let kirchhoff-largest = calc.max(..kirchhoff.map(row => float(row.max_relative_residual)))
#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    table.header(
      table.cell(rowspan: 2)[Case],
      table.cell(colspan: 2, align: center)[Uniform], table.cell(colspan: 2, align: center)[Targeted],
      table.cell(colspan: 2, align: center)[Gate-bias],
      [Median], [Max], [Median], [Max], [Median], [Max],
    ),
    ..case-names.keys().map(case-name => (
      case-names.at(case-name),
      ..("uniform", "targeted", "gatebias").map(set-name => {
        let row = kirchhoff.find(row => row.case == case-name and row.set == set-name)
        (sci(row.median_relative_residual, digits: 1),
          table.cell(fill: if float(row.max_relative_residual) == kirchhoff-largest { warn-light } else { none },
            sci(row.max_relative_residual, digits: 1)))
      }).flatten(),
    )).flatten(),
  ),
  caption: [The four currents of every sample sum to zero within the printed precision. The residual is |Id + Ig + Is + Ib| divided by the largest terminal current of the sample. The largest value is shaded.],
) <tab-kirchhoff>

The second check is that the temperature and the process values of a sample take effect in the
simulation. @fig-trends simulates one transistor with the same netlist code as the samples and
changes only the quantity on the horizontal axis. The OFF drain current grows by 2.6 decades from
-40 to 125 °C (left) and falls by 1.5 to 1.7 decades as |vth0| goes from 10 % below to 10 % above
its nominal value (middle). The ON gate current falls by 1.5 decades as the oxide thickness toxe
goes from 10 % below to 10 % above nominal (right). Each current changes in the direction that
device physics predicts.

#figure(
  placement: auto,
  image("/results/figures/trend_sweep.pdf", width: 100%),
  caption: [The simulated currents follow device physics. Each panel is a SPICE sweep of one quantity on a transistor of width 4 Lmin, with everything else at its nominal value (1 V, 27 °C).],
) <fig-trends>

== Split

We split each set of each case into training, validation and test samples, 70/15/15, with
seed 463. So the 10,000 samples of the uniform set split 7,000/1,500/1,500 and the 5,000 of the
targeted set 3,500/750/750. The models train on the uniform and targeted training samples together,
7,000 + 3,500 = 10,500 per case. Errors are reported for each set separately.

= Transistor models <sec-models>

== Inputs, targets and preprocessing

A leakage current is hard to fit as it is, because it varies over many orders of magnitude and Id and Is change sign (@fig-currents). So we give the models features in which the current is smooth, and a target that compresses its range.

*Features.* Each model computes 20 features from the 12 inputs of a sample. Eight are raw inputs
(VDD, Vd, Vs, |vth0|, toxe, ndep, u0 and xj), and seven are differences of the terminal voltages
(Vds, Vgd, Vgs, Vdb, Vsb, and Vgs and Vbs against the effective source). Five follow the BSIM4 form
of the leakage currents, with v#sub[T] = kT/q.
- |Vds| and log(1 - exp(-|Vds| / v#sub[T])), the drain factor of the subthreshold current
- (Vgs - |vth0|) / v#sub[T], its exponent
- 1/T and log W
BSIM4 is symmetric in drain and source, so Vgs and Vbs are taken against the lower of Vd and Vs for
NMOS and the higher for PMOS. All features are standardised with the mean and standard deviation of
the training samples.

*Targets.* One model predicts all four terminal currents of its case. In the uniform set the drain
current of an ON NMOS lies between 1.9e-19 A and 1.6e-4 A, a factor of about 10#super[15]. A model
trained on the current itself would fit only the largest currents, because a 1 % error at 1e-4 A is
1e-6 A, more than the whole current of a sample at 1e-7 A. We want the same relative error on every
sample, so the target is logarithmic, where a 1 % error has the same size at every current. We
tested two such targets.
- log: log#sub[10]|I| of each terminal. A current of exactly zero has no logarithm, so any |I| below
  1e-30 A is set to 1e-30 A before the logarithm is taken. The sign comes from a classifier of the
  same model type, for the terminals whose sign changes in the training samples or that are exactly
  zero in some of them (in practice Id and Is, and Ig in the OFF cases).
- asinh: asinh(I / I#sub[0]), with I#sub[0] the 1st percentile of the nonzero |I| of the training samples
  per case and terminal. It is linear below I#sub[0] and logarithmic above it, so the sign change
  passes smoothly through zero.

#figure(
  placement: auto,
  image("/report/analysis/current_ranges.pdf", width: 100%),
  caption: [The currents a model has to predict differ by many orders of magnitude. The bar covers the 5th to 95th percentile of |I| in one sample set, the thin line its full range, and the white tick its median. Currents below 1e-30 A, the smallest value our log target takes, are drawn at the left edge. The dashed line marks 1e-26 A.],
) <fig-currents>

*Metrics.* Every model is scored on the relative error |I#sub[ML] - I#sub[SPICE]| / |I#sub[SPICE]| and on R² of
log#sub[10]|I|. An R² on the current itself would only measure the largest currents.

== Four model types

We tried four model types, each with one set of hyperparameters (@tab-hyper), on both
targets. @tab-comparison compares them on the validation samples, with four rows per case, one per
model type. Each model predicts four terminal currents, so every number is the mean over the four.
The _Score_ columns give the quantity we select on, once for each target. The score is the median
error averaged over the terminals and over the uniform and targeted sets. The shaded cell is the
lowest score of a case.

#figure(
  small-table(table(
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    table.header([Model], [Main hyperparameters], [Sign classifier (log target)]),
    [Poly ridge], [Degree-3 polynomial features, standardised, `RidgeCV` with α from 1e-8 to 1e2 (11
      values), one α per output], [Logistic regression on the same features],
    [Extra trees], [300 trees, scikit-learn defaults otherwise], [Extra-trees classifier, 300 trees],
    [LightGBM], [2,000 rounds, learning rate 0.03, 31 leaves, at least 10 samples per leaf, one regressor
      per terminal], [LightGBM classifier, 500 rounds],
    [MLP], [4 hidden layers of 256 SiLU units, Adam with a one-cycle schedule up to 2e-3, batch 512,
      2,000 epochs, mean squared error, on the GPU], [The same network with a logistic loss],
  )),
  caption: [The four model types and their hyperparameters. Every model sees the same 20 standardised features.],
) <tab-hyper>

#let comparison = load("/report/analysis/model_summary.csv")
#let comparison-row(case-name, model, transform) = comparison.find(row => row.case == case-name
  and row.model == model and row.transform == transform)
#figure(
  placement: auto,
  small-table(table(
    columns: 8,
    align: (left, left, right, right, right, right, right, right),
    inset: 4pt,
    table.header(
      table.cell(rowspan: 2)[Case], table.cell(rowspan: 2)[Model],
      table.cell(colspan: 3, align: center)[Uniform validation, log target],
      table.cell(rowspan: 2)[Targeted median (%)~#good-down],
      table.cell(colspan: 2, align: center)[Score (%)~#good-down],
      [R²~#good-up], [Median (%)~#good-down], [p95 (%)~#good-down], [log], [asinh],
    ),
    ..case-names.keys().map(case-name => {
      let best = calc.min(..comparison.filter(row => row.case == case-name and row.model != "null_median")
        .map(row => float(row.selection_score)))
      let score(row) = table.cell(
        fill: if float(row.selection_score) == best { accent-light } else { none },
        pct-of(row.selection_score),
      )
      let models = ("poly_ridge", "extra_trees", "lightgbm", "mlp")
      let log-rows = models.map(model => comparison-row(case-name, model, "log"))
      // The best value of a column within the case is bold.
      let mark(row, key, pick, body) = if float(row.at(key)) == pick(..log-rows.map(other => float(other.at(key)))) {
        strong(body)
      } else { body }
      models.enumerate().map(((index, model)) => {
        let log-row = log-rows.at(index)
        let cells = (
          model-names.at(model),
          mark(log-row, "uniform_r2_log10", calc.max, fixed(log-row.uniform_r2_log10, 3)),
          mark(log-row, "uniform_median_rel_error", calc.min, pct-of(log-row.uniform_median_rel_error)),
          mark(log-row, "uniform_p95_rel_error", calc.min, pct-of(log-row.uniform_p95_rel_error)),
          mark(log-row, "targeted_median_rel_error", calc.min, pct-of(log-row.targeted_median_rel_error)),
          score(log-row),
          score(comparison-row(case-name, model, "asinh")),
        )
        if index == 0 { (table.cell(rowspan: 4, case-names.at(case-name)),) + cells } else { cells }
      }).flatten()
    }).flatten(),
  )),
  caption: [The four model types on the validation samples. Each number is the mean over the four terminal currents. R² is on log#sub[10]|I|, p95 is the 95th percentile of the error, and the score is the quantity the selection uses. An arrow says whether higher or lower is better. The best value of each column within a case is bold, and the lowest score of each case is shaded.],
) <tab-comparison>

The shaded cell is in an MLP row in all four cases, so the MLP is selected everywhere. On the
uniform validation samples its median error is about ten times lower than that of the next best
type, LightGBM, in the ON cases and about two times lower in the OFF cases. We attribute this margin
to the form of the currents. They are exponential in the voltages, the temperature and vth0, so
log#sub[10]|I| is a smooth function of the features. An MLP fits a smooth function, while trees
return a piecewise-constant one, which leaves an error set by how finely 10,500 samples cut the
input space. The degree-3 polynomial cannot follow the curvature over the whole input range.

The log target has the lower score in 13 of the 16 combinations of model type and case. One likely
reason is the sign. With asinh the network has to pass Id and Is through zero inside the narrow band
of Vds near 0, a steep step that it smooths over, while with the log target a separate classifier
carries the sign. The one exception for the MLP is NMOS OFF, where asinh is lower by less than a
point. This table does not settle a difference that small, and we use the log target in every case
from here on.

The MLP is not the best model on every metric. In the OFF cases LightGBM has the highest R² (bold).
The whole R² gap is the OFF gate current, the smallest currents of @fig-currents, which the MLP fits
poorly. On the other three terminals its R² is 0.996 or above.

To test whether fewer samples would suffice, we retrain the model with the lowest score of each case on the first
1,500, 3,000 and 7,500 training samples, with both sets in proportion (@fig-learning). The error
falls at every step in all four cases and is still falling at 10,500, so we kept all the samples.

#figure(
  placement: auto,
  image("/report/analysis/learning_curve.pdf", width: 56%),
  caption: [The error still falls as training samples are added. Each line is the median error on the uniform validation samples, as a mean over the four terminals, of the model with the lowest score of the case in @tab-comparison. For NMOS OFF that is the MLP with the asinh target, while every later step uses the log target.],
) <fig-learning>

== From the baseline to the final models <sec-selection>

The MLP with the log target, one per case, is our baseline. Used for the gates as in @sec-gates it
reaches 2.07 % mean gate error on the 200 corners of Part B, twice the 1 % we aim for. The Part B
results showed where the baseline fails, as below, but no model was trained or picked on Parts B or
C.

*How a change is judged.* The score that picked the baseline misses two things. It averages in the
ON drain and source currents, which the gate leakage never uses because it gets them from
Kirchhoff's current law (@sec-compose), and it is measured on uniform and targeted samples, not at
the voltages a transistor has in a gate. For the baseline the NMOS ON gate current was off by 0.65 %
on the targeted test samples but by 2.0 % in the Part B gates. So we judged every change on the
gate-bias set of @tab-sets. The gate-bias score is
sum |I#sub[ML] - I#sub[SPICE]| / sum |I#sub[SPICE]| over the currents the gate leakage takes from
the models, Ig and Ib of the ON cases and all four currents of the OFF cases, on the gate-bias
validation samples, averaged over the four cases. A gate's leakage is a sum of such currents, so
each one counts by its size. The score also varies between two trainings of the same method, and
this variation is the grey band of @fig-arms. Before any run we wrote down, for each change, the
result that would make us drop it, and a metric counted as improved only if it beat the baseline by
more than this variation (@sec-protocol).

#figure(
  placement: auto,
  image("/report/analysis/arm_scores.pdf", width: 88%),
  caption: [Which changes improve the transistor model. Each point is the gate-bias score of one training run, where lower is better, and the short line is the mean of its group. The grey band is the baseline mean ± 2 standard deviations, so a mean below the band is an improvement larger than the variation between trainings. The grey points belong to the two changes we dropped, and the red points are the final method.],
) <fig-arms>

In @fig-arms each column is one method and each point is one training. A method whose points lie
below the grey band is better than the baseline by more than the variation between trainings. We
tested five changes, A1 to A5. A1 and A2 stay inside the band, and A3, A4 and A5 lie below it. The
two columns to the right of the dashed line are combinations. We kept three changes, and
@fig-variants draws the model with them.

*A second MLP for the ON gate and body currents (A2).* The baseline fits all four currents of an ON
transistor with one network, although the gate leakage uses only Ig and Ib. A2 gives those two
currents an MLP of their own, so that they no longer share weights with Id and Is. The error of the
ON gate current fell to less than a third. On the score as a whole A2 alone stays inside the band.
It is kept because it met its own condition, which is on the ON gate current (@tab-arms).

*The smallest OFF gate currents left out of the loss (A4).* The gate current of an OFF transistor
is the smallest current of @fig-currents. In the targeted and gate-bias sets 35 to 38 % of the OFF
samples are at or below 1e-26 A, the dashed line of the figure, and some are exactly zero. These
currents do not matter for a gate, whose leakage is at least 1.9e-12 A in Part B. With a log target
they still count. The step from 1e-30 to 1e-26 A is as large as the step from 1e-16 to 1e-12 A, so
a model spends part of its fit on them. A4 leaves them out of the loss. The network then fits the
OFF drain current better, and A4 lowers the score more than any other single change.

*An ensemble of five (A5).* Two fits of the same model differ, as the spread of the baseline points
shows. A5 trains five copies with seeds 463 to 467 and averages them in log#sub[10]|I|, with a
majority vote on the sign.

We dropped the other two changes. Fitting the ON gate current per effective width (A1) did not help
the narrow transistors it was meant for. Training on the gate-bias set itself (A3) lowered the
score, but it left the OFF drain current across the rails less accurate than the limit we had set
(@sec-protocol).

#figure(
  placement: auto,
  image("/figures/architecture/model_variants.pdf", width: 100%),
  caption: [The transistor model of one case in three stages. On the left is the baseline, one MLP for the four currents and a classifier for the sign of Id and Is (and of Ig in the OFF cases). The middle adds a second MLP for the gate and body currents of the ON cases, and a mask that leaves the smallest OFF gate currents out of the loss. On the right is the final model, which averages five copies of the middle one trained with different seeds.],
) <fig-variants>

*The final model.* A2 and A4 together score lower than any single change, and the ensemble on top
lowers the score again. The final model is this combination. Each case is still one model, one saved
file with one predict call, in the way an extra-trees model is one model that averages 300 trees.

== Final models on the test samples

@tab-chosen gives the errors of the final models on the test samples, for each case, sample set and
terminal current. On the uniform test samples (shaded in blue) the ON cases are more accurate than
the OFF cases, and both improve on the baseline, to roughly half its median error for the ON cases
and a third for the OFF cases.

#figure(
  placement: auto,
  small-table(table(
    columns: 10,
    align: (left, left, right, right, right, right, right, right, right, right),
    inset: 4pt,
    table.header(
      table.cell(rowspan: 2)[Case], table.cell(rowspan: 2)[Set],
      table.cell(colspan: 4, align: center)[Median error (%)~#good-down], table.cell(colspan: 4, align: center)[95th-percentile error (%)~#good-down],
      [Id], [Ig], [Is], [Ib], [Id], [Ig], [Is], [Ib],
    ),
    ..load("/report/analysis/chosen_summary.csv").map(row => {
      let on = row.case.ends-with("_on")
      let uniform = row.set == "uniform"
      // Orange: the two groups of large errors of the text, the ON Id and Is and the OFF Ig outside
      // the uniform set.
      let fill(terminal) = if uniform { accent-light } else if (on and terminal in ("id", "is")) or (not on and terminal == "ig") { warn-light } else { none }
      (
        table.cell(fill: fill(""), case-names.at(row.case)),
        table.cell(fill: fill(""), (uniform: "uniform", targeted: "targeted", gatebias: "gate-bias").at(row.set)),
        ..("median_rel_error", "p95_rel_error").map(key => ("id", "ig", "is", "ib").map(terminal =>
          table.cell(fill: fill(terminal), pct-of(row.at(terminal + "_" + key))))).flatten(),
      )
    }).flatten(),
  )),
  caption: [Errors of the final models on the test samples of each sample set, for each terminal current. The uniform test samples are shaded in blue. The cells shaded in orange are the two groups of large errors, which the gate leakage does not depend on.],
) <tab-chosen>

Two groups of currents have large errors in the table (shaded in orange), and neither affects the
gate leakage. The first is the ON Id and Is on the targeted and gate-bias samples. Near Vds = 0
these currents change sign, and the sign classifier is wrong in 10 to 12 % of the targeted and 38 to
39 % of the gate-bias test samples. A wrong sign costs about 200 %, which is the 95th percentile of
the targeted samples in the table. The gate leakage does not use these currents. The second is the
OFF Ig on the targeted and gate-bias samples. Many of these currents are at or below 1e-26 A, more
than 14 decades below the smallest gate leakage of Part B, and A4 leaves those out of the loss.

@fig-parity plots the same test samples, with the SPICE current on the horizontal axis and the predicted current on the vertical axis, both logarithmic. In most panels the points lie on the diagonal over the full range of the current. The two groups above are the visible exceptions. The samples with a wrong sign are marked in the Id and Is panels, and the Ig panels of the two OFF cases show a wide cloud at the smallest currents.

#figure(
  placement: auto,
  stack(
    spacing: 2pt,
    image("/results/figures/parity_nmos_on.pdf", width: 84%),
    image("/results/figures/parity_nmos_off.pdf", width: 84%),
    image("/results/figures/parity_pmos_on.pdf", width: 84%),
    image("/results/figures/parity_pmos_off.pdf", width: 84%),
  ),
  caption: [Predicted against SPICE current for the final models on the test samples. The rows are, from the top, NMOS ON, NMOS OFF, PMOS ON and PMOS OFF, and the columns are the four terminal currents. A point on the diagonal is an exact prediction. Currents that are exactly zero are drawn at 1e-30 A.],
) <fig-parity>

= Gate results <sec-gates>

== Gates, corners and SPICE reference <sec-reference>

The five gates are static CMOS (@tab-gate-sizes). A gate with one input combination is a gate case,
written as the gate and its inputs, for example NAND2 10. We draw 200 corners as in @tab-uniform. A
corner is one VDD, one temperature, one NMOS process set and one PMOS process set shared by all
transistors of that type. Every gate case runs at every corner with the inputs at exactly 0 or VDD,
18 × 200 = 3,600 simulations. For each one we record the leakage, the output voltage and the
internal node voltages.

#figure(
  placement: none,
  small-table(table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, right),
    table.header([Gate], [NMOS width (Lmin)], [PMOS width (Lmin)], [Gate cases]),
    [INV], [1], [2], [2],
    [NAND2], [2, in series, B at the output], [2], [4],
    [NOR2], [1], [4, in series, A at the supply], [4],
    [AND2], table.cell(colspan: 2)[NAND2 followed by the INV, one netlist], [4],
    [OR2], table.cell(colspan: 2)[NOR2 followed by the INV, one netlist], [4],
  )),
  caption: [The five gates and their 18 gate cases.],
) <tab-gate-sizes>

The SPICE leakage is the sum, over the nodes held at VDD, of the terminal currents ngspice reports
for each transistor. An input at VDD counts because it drives the gate tunnelling current of the
transistors it controls, and in a circuit that current comes from the supply of the gate driving the
input. To judge a prediction to better than 1 % this reference has to be far more accurate than
that, and we checked it in three ways (@sec-reference-checks).

== From transistor currents to a gate leakage <sec-compose>

For the ML prediction each transistor of each simulation is one model input with the corner's
VDD, temperature and process set, its width, and its terminal voltages from the SPICE node voltages. Its gate voltage picks the case. An NMOS with its gate above VDD/2 is ON,
and a PMOS with its gate below VDD/2 is ON. So the ML prediction uses the SPICE node voltages and has none of its own.

The direct way to obtain a gate leakage from the predicted currents, which we call the plain sum, adds up every predicted current on the nodes held at VDD. For the NAND2 of @fig-composition those are the outlined currents, three of the OFF PMOS mpa, two of the ON PMOS mpb and the gate current of the ON NMOS mna. The plain sum fails whenever the leakage leaves the supply through an ON transistor, here mpb. That transistor has microvolts across it, and its
source current is whatever the OFF network below it draws. Its own terminal voltages say almost nothing about it.
The simplest case is the INV with A = 0. The supply delivers the source and body currents of the ON
PMOS (mp), and current balance at the output node gives

$ I_"leak" = I_(s,"mp") + I_(b,"mp") = I_(d,"mn") - I_(g,"mp") $

since I#sub[d,mp] = -I#sub[d,mn] and the four currents of mp sum to zero. So the leakage is the OFF
NMOS drain current plus the gate tunnelling of the ON PMOS, two currents the models predict well,
instead of the ON PMOS source current at 1 µV, which they do not.

We apply the same current balance to every gate and call the result the Kirchhoff sum, which is the method we use. Every OFF transistor keeps all four predicted currents, and every ON
transistor keeps its predicted gate and body currents (yellow in @fig-composition). The drain
currents of the ON transistors (blue) are the unknowns of one linear system per gate case,
with one equation per node that no source holds (out and x in the figure), saying that the currents into it sum to
zero. It is solved by least squares. An ON source current is then -(I#sub[d] + I#sub[g] + I#sub[b]).
Parallel ON transistors share a node pair, so least squares splits their current equally, and only
their sum reaches the leakage.

As a control we replaced the models by each transistor's own SPICE currents and ran the same code.
It gives back the SPICE leakage to within rounding (@sec-code-checks). So any error below comes from
the models and not from the Kirchhoff sum.

== Results

@tab-gates gives the result for each of the 18 gate cases over the 200 corners of Part B, and for
all 3,600 simulations in its last row (shaded in blue). The error of one simulation is
(ML - SPICE) / SPICE. The _Bias_ column is the mean of this signed error. The mean, median, 95th
percentile and maximum are of its magnitude. Over all simulations the mean error is
#fixed(gate-all.mean_error_percent, 2) %, below the 1 % we aim for and less than a third of the
2.07 % of the baseline.

The last column of the table is the median error of the plain sum, for comparison. The plain sum is
identical to the Kirchhoff sum in the four cases where no ON transistor has its source on the supply
(INV 1, NAND2 11, NOR2 10 and 11). In the other 14 an ON PMOS takes the leakage from the supply, and
the plain sum underestimates it by tens of percent. In some cases it misses by more than the leakage
itself, which gives a negative prediction.

@fig-gate-parity shows the same comparison for every simulation. Each point is one simulation, with
the SPICE leakage on the horizontal axis and the predicted leakage on the vertical axis, both
logarithmic. With the Kirchhoff sum (left) the points lie on the diagonal over the whole range of
leakage. With the plain sum (right) most gates fall below the diagonal.

#let gate-medians = gate-ml.filter(row => row.gate != "all").map(row => float(row.median_error_percent)).sorted()
#figure(
  placement: auto,
  small-table(table(
    columns: 10,
    align: (left, left, right, right, right, right, right, right, right, right),
    inset: (x: 5pt, y: 3pt),
    table.header([Gate], [Inputs], [SPICE median (pA)], [ML median (pA)], [Bias (%)], [Mean (%)~#good-down],
      [Median (%)~#good-down], [p95 (%)~#good-down], [Max (%)~#good-down], [Plain-sum median (%)~#good-down]),
    ..gate-ml.zip(gate-terminal).map(((row, terminal)) => {
      let cells = (
        if row.gate == "all" [All] else [#gate-name(row.gate)], if row.inputs == "all" [] else [#row.inputs],
        fixed(1e12 * float(row.spice_median_A), 1), fixed(1e12 * float(row.predicted_median_A), 1),
        fixed(row.bias_percent, 2),
        fixed(row.mean_error_percent, 2),
        // The three lowest medians are bold and the three highest shaded in orange.
        if row.gate == "all" { fixed(row.median_error_percent, 2) }
        else if float(row.median_error_percent) <= gate-medians.at(2) { strong(fixed(row.median_error_percent, 2)) }
        else if float(row.median_error_percent) >= gate-medians.at(-3) { table.cell(fill: warn-light, fixed(row.median_error_percent, 2)) }
        else { fixed(row.median_error_percent, 2) },
        fixed(row.p95_error_percent, 2),
        fixed(row.max_error_percent, 2), pct-wide(terminal.median_error_percent),
      )
      if row.gate == "all" { cells.map(body => table.cell(fill: accent-light, body)) } else { cells }
    }).flatten(),
  )),
  caption: [Gate leakage of the final models against SPICE for each gate and input combination, over 200 corners each. Mean, median, p95 and max are of the error |ML - SPICE| / SPICE with the Kirchhoff sum. The last column is the median error of the plain sum. The three lowest medians are bold and the three highest are shaded in orange.],
) <tab-gates>

#figure(
  placement: auto,
  block(width: 74%, grid(
    columns: 2,
    gutter: 8pt,
    image("/results/figures/gate_parity_ml.pdf"),
    image("/results/figures/gate_parity_ml_terminal.pdf"),
  )),
  caption: [Predicted against SPICE gate leakage for all 3,600 simulations. On the left is the Kirchhoff sum, where the points lie along the diagonal. On the right is the plain sum, which falls below the diagonal for most gates. Its negative predictions cannot be drawn on log axes and are missing.],
) <fig-gate-parity>

The median column of @tab-gates shows which cases are most and least accurate. The most accurate are
the three NAND2 cases with an input at 0 (bold). The three least accurate are NOR2 11, NOR2 10 and
INV 1 (shaded in orange). They are the only cases with a median above 1 %, about twice the next
case, and they also have the largest biases, all positive. All three have the output low and an
input at VDD on an NMOS of width 1 Lmin. @sec-errors shows that their error is the predicted gate
current of that ON NMOS.

= Test-circuit results <sec-adder>

The adder tests whether gate leakages can simply be added up. It uses only gates of Part B, NAND2,
AND2 and OR2, with each XOR built from four NAND2 and the carries from AND2 and OR2.

$ P_i = A_i xor B_i, quad G_i = A_i B_i, quad S_i = P_i xor C_i $
$ C_1 = G_0 + P_0 C_0, quad C_2 = G_1 + P_1 G_0 + P_1 P_0 C_0, quad
  C_3 = G_2 + P_2 G_1 + P_2 P_1 G_0 + P_2 P_1 P_0 C_0 $

Here C#sub[0] = Cin and Cout = C#sub[3]. That is the 42 gates of @fig-composition (24 NAND2, 12 AND2
and 6 OR2), 204 transistors in one flat netlist. With 7 inputs there are 128 vectors, so the SPICE
reference is 128 × 200 = 25,600 simulations at the corners of Part B. The netlist computes
A + B + Cin for all 128 vectors, and every gate sees the inputs our code assumes
(@sec-adder-checks).

The adder leakage is built from the gates. For each vector we find the input combination of every
gate and sum the Part B gate leakages of those cases at the same corner. To measure the error of
adding up separately simulated gates, before any model enters, we summed the SPICE gate leakages of
Part B instead of the ML ones as a control. That sum agrees with the SPICE adder to a few
thousandths of a percent (first row of @tab-cla). So the loading between gates, and inputs a few
microvolts off the rails, cost almost nothing here, and the adder error is the gate error.

#figure(
  placement: auto,
  small-table(table(
    columns: 8,
    align: (left, right, right, right, right, right, right, right),
    inset: (x: 5pt, y: 3pt),
    table.header([Gate leakages summed], [SPICE median (nA)], [Sum median (nA)], [Bias (%)], [Mean (%)~#good-down],
      [Median (%)~#good-down], [p95 (%)~#good-down], [Max (%)~#good-down]),
    ..(
      ([SPICE gate leakages (control)], cla-control),
      ([From SPICE transistor currents (control)], cla-oracle),
      ([ML, Kirchhoff sum], cla-ml),
      ([ML, plain sum], cla-terminal),
    ).map(((label, row)) => {
      let cells = (
        label, fixed(1e9 * float(row.spice_median_A), 3), fixed(1e9 * float(row.predicted_median_A), 3),
        pct-wide(row.bias_percent),
        pct-wide(row.mean_error_percent), pct-wide(row.median_error_percent), pct-wide(row.p95_error_percent),
        pct-wide(row.max_error_percent),
      )
      if row == cla-ml { cells.map(body => table.cell(fill: accent-light, body)) } else { cells }
    }).flatten(),
    table.hline(stroke: 0.5pt + luma(120)),
    [Gates alone, ML, Kirchhoff sum], fixed(1e9 * float(gate-all.spice_median_A), 4),
    fixed(1e9 * float(gate-all.predicted_median_A), 4),
    fixed(gate-all.bias_percent, 2), fixed(gate-all.mean_error_percent, 2), fixed(gate-all.median_error_percent, 2),
    fixed(gate-all.p95_error_percent, 2), fixed(gate-all.max_error_percent, 2),
  )),
  caption: [Adder leakage against the SPICE adder for 25,600 simulations at the 200 corners of Part B. Each row adds up gate leakages obtained in a different way. The first two rows are controls that use SPICE values, and the next two use the ML gate leakages. Mean, median, p95 and max are of the error. The last row repeats the gates of @tab-gates for comparison. The result is shaded.],
) <tab-cla>

In @tab-cla each row sums gate leakages obtained in a different way and compares the sum with the
SPICE adder. The first two rows are the controls with SPICE values, and the shaded row is the ML
adder with the Kirchhoff sum. Its mean error is #fixed(cla-ml.mean_error_percent, 2) %, and in
@fig-cla all 25,600 simulations lie on the diagonal.

The last row of the table repeats the gates for comparison. The adder is more accurate than the
gates alone in every error column. It is expected to be somewhat worse, since its gates load each
other and their inputs sit slightly off the rails, but the control above shows that this effect is
negligible. The adder is more accurate for two reasons. It is built only from NAND2, AND2 and OR2,
whose 12 cases have a mean error of 0.39 % against 0.58 % for all 18, because the least accurate
cases are the standalone INV and NOR2 (@tab-gates). And it is a sum of 42 gate leakages, so errors
of opposite sign in different gate cases partly cancel. The cancelling is limited, since the gate
errors of one corner mostly share a sign (@sec-errors). No vector is harder than the others. The
median error per vector lies between 0.14 and 0.16 % over all 128, so the corner matters far more
than the vector.

*Time.* One SPICE run of the adder takes #fixed(timing("spice_seconds_per_adder_leakage"), 2) s on
average, in a single process on the machine of @tab-setup. Our method needs the 18 gate cases of a
corner in SPICE, #fixed(1000 * timing("spice_gate_seconds_per_run"), 1) ms each, and
#fixed(1000 * timing("ml_seconds_per_corner"), 0) ms for the models and the sums. That gives the
leakage of all 128 vectors of the corner, so one adder leakage costs
#fixed(1000 * timing("method_seconds_per_adder_leakage"), 1) ms on average, about #speedup times less
than SPICE. The 18 gate cases are the same for every vector, so this cost per corner does not grow with
the number of vectors. For a single gate there is no saving,
because the prediction uses the node voltages of the SPICE run of that gate. The time to simulate
the training samples and to train the models is paid once and is not counted.

#figure(
  placement: auto,
  image("/results/figures/cla_parity_ml.pdf", width: 36%),
  caption: [Predicted against SPICE adder leakage for all 25,600 simulations, with the Kirchhoff sum.],
) <fig-cla>

= Validation on held-out corners <sec-blind>

The 200 corners of Parts B and C were in front of us while we developed the method, so they cannot
also test it. We therefore tested the final method once on 400 new corners, drawn only after the
models were final and independent of the 200 of Part B. We call them the held-out corners. The
corners, the statistics, the pass criteria and the controls were written down before any model ran
on them (@sec-statistics).

On these corners we ran the gates and the adder in SPICE, the final models, the baseline models and
two controls. The first control replaces the models by the SPICE transistor currents, as in
@sec-compose. The second always predicts the median Part B leakage of each gate case or adder
vector, so it knows nothing about the corner.

#let blind-label = (frozen: [Final models], old: [Baseline models], oracle: [SPICE currents],
  null: [Part B median])
#figure(
  placement: auto,
  small-table(table(
    columns: 8,
    align: (left, left, right, right, right, right, right, right),
    inset: (x: 5pt, y: 3pt),
    table.header([Circuit], [Predictor], [Mean (%)~#good-down], [95 % interval of the mean], [Median (%)~#good-down],
      [p95 (%)~#good-down], [Max (%)~#good-down], [R² (log#sub[10])~#good-up]),
    ..blind.filter(row => row.method != "frozen minus old").map(row => {
      let cells = (
        if row.level == "gate" [Gates] else [Adder], blind-label.at(row.method),
        pct-wide(row.mean_error_percent), [#pct-wide(row.ci_low) to #pct-wide(row.ci_high)],
        pct-wide(row.median_error_percent), pct-wide(row.p95_error_percent), pct-wide(row.max_error_percent),
        if row.r2_log10 == "" [] else { fixed(row.r2_log10, 5) },
      )
      if row.method == "frozen" { cells.map(body => table.cell(fill: accent-light, body)) } else { cells }
    }).flatten(),
  )),
  caption: [The final method on 400 new corners, which is 7,200 gate and 51,200 adder simulations. The rows _SPICE currents_ and _Part B median_ are the two controls. Mean, median, p95 and max are of the error.],
) <tab-blind>

@tab-blind has one block of rows for the gates and one for the adder. In each block the shaded row
is the final models, followed by the baseline models and the two controls. Every criterion we had
set holds.
- Below 1 %. The whole 95 % interval of the mean error is below 1 %, for the gates and for the adder.
- Better than the baseline. On the same corners the final models have less than a third of the
  baseline's mean error for the gates and less than a fifth for the adder, and the 95 % interval of
  each difference is entirely below zero.
- Large errors did not grow. The 95th percentile and the maximum are also lower than for the
  baseline.
- Controls. With SPICE currents in place of the models the gate error is at the level of rounding,
  and the adder error is 0.003 %, the cost of adding up gates. The
  Part B median, which knows nothing about the corner, is off by more than 80 % in the mean. So the
  evaluation tells a model from no model.

The 200 corners of Part B give the same picture. Their mean errors (@tab-gates, @tab-cla) are within
0.03 points of the held-out ones. So the work on Part B did not fit the method to those corners.

= Error analysis <sec-errors>

To explain the errors on the 200 corners of Part B we ran the final models again on the 16,800 transistors of the Part B gates and kept every predicted current. The gate errors computed this way match @tab-gates to 4e-6 points.

== Contribution of each transistor model

@fig-attribution shows which transistor model the gate error comes from. The grey bar is the median
error of a gate case with all four transistor models. For each marker we repeat the calculation with
the ML currents of one transistor case only and the SPICE currents of the other three. A marker near
the top of its bar therefore means that this one model explains the error of that gate case. In the
three worst cases, INV 1, NOR2 10 and NOR2 11, the filled blue marker of the NMOS ON model sits
almost at the top of the bar. The open markers of the OFF models stay in the lower half of the bar
except in INV 0 and NOR2 00, where the NMOS OFF model explains most of the error.

#figure(
  placement: auto,
  image("/report/analysis/attribution.pdf", width: 100%),
  caption: [The ON NMOS model causes the largest gate errors. The grey bar is the median error of each gate and input combination with all four transistor models. Each marker is the error when only one transistor case uses its model and the other three use their SPICE currents.],
) <fig-attribution>

The ON models dominate because a gate's leakage in this technology is mostly gate tunnelling. The
gate current of the ON transistors is 97.5 % of the leakage over all simulations. The three cases
stand out because their ON NMOS are 1 Lmin wide, the lower end of the sampled width range. At that
width the median error of the ON NMOS gate current is 1.2 %, nearly all of it over-prediction,
against 0.2 % with almost no bias at 2 Lmin. With W uniform in 1 to 8 Lmin, about 7 % of the training samples lie below 1.5 Lmin, and
such a sample has other training samples only on its wider side. The bias at 1 Lmin reappears as the
positive bias of the three cases in @tab-gates. NAND2 11 also has two ON NMOS, but at 2 Lmin, and it
is not among the least accurate cases. We tried to fix this by modelling the gate current per
effective width (change A1), and it did not help.

== ON transistors at microvolts

@tab-transistor-errors tests whether the targeted set fixed the ON transistors with microvolts
across them (@sec-gap). Each row is one current of the transistors of the Part B gates, and the
last two columns give its median error for models trained on the uniform set alone and for the final
models. The lower of the two is bold. The first six rows are the ON transistors with |Vds| below
1 mV, and the last two are the OFF drain currents.

#let transistor-errors = load("/report/analysis/transistor_errors.csv")
#let transistor-error(models, case-name, small, terminal) = {
  let found = transistor-errors.filter(row => row.models == models and row.case == case-name
    and row.vds_below_1mV == small and row.terminal == terminal)
  pct-of(found.at(0).median_rel_error)
}
#figure(
  placement: auto,
  small-table(table(
    columns: 6,
    align: (left, left, left, right, right, right),
    inset: (x: 6pt, y: 3.5pt),
    table.header([Case], [|Vds|], [Current], [Transistors], [Uniform-only models (%)~#good-down], [Final models (%)~#good-down]),
    ..(
      ("nmos_on", "True", "ig"), ("nmos_on", "True", "is"), ("nmos_on", "True", "id"),
      ("pmos_on", "True", "ig"), ("pmos_on", "True", "is"), ("pmos_on", "True", "id"),
      ("nmos_off", "False", "id"), ("pmos_off", "False", "id"),
    ).map(((case-name, small, terminal)) => {
      let found(models) = transistor-errors.filter(row => row.models == models and row.case == case-name
        and row.vds_below_1mV == small and row.terminal == terminal).at(0)
      // The lower error of the two columns is bold.
      let shown(models, other) = if float(found(models).median_rel_error) < float(found(other).median_rel_error) {
        strong(transistor-error(models, case-name, small, terminal))
      } else { transistor-error(models, case-name, small, terminal) }
      let cells = (case-names.at(case-name), if small == "True" [below 1 mV] else [1 mV or more],
        upper(terminal.at(0)) + terminal.slice(1), found("chosen").rows,
        shown("uniform only", "chosen"), shown("chosen", "uniform only"))
      // The gate currents are the ON currents the gate leakage takes from the models.
      if terminal == "ig" { cells.map(body => table.cell(fill: accent-light, body)) } else { cells }
    }).flatten(),
  )),
  caption: [Median error of the predicted currents of the transistors of the Part B gates, for models trained on the uniform set alone and for the final models. The lower error of each row is bold, and the gate-current rows are shaded.],
) <tab-transistor-errors>

The shaded rows show the effect. The gate current of the ON transistors goes from 41 % (NMOS) and
116 % (PMOS) with the uniform-only models to below 1 % with the final models. Most of this comes
from the targeted set. The baseline, which adds it, is at 2.0 % (NMOS) and 1.1 % (PMOS), and the
gate and body MLP gave the rest. The ON drain and source currents stay far off in both columns, and the Kirchhoff sum is
what makes that harmless.

The targeted set cost something on the OFF side (last two rows). In the baseline the OFF drain
current in the Part B gates was less accurate than with the uniform-only models. Leaving the
smallest OFF gate currents out of the loss, together with the ensemble, won most of that back. The
PMOS is now better than without the targeted set and the NMOS is still worse.

== Corner conditions

The left panel of @fig-corner plots the error of every gate simulation against the temperature of
its corner, with the error on a logarithmic axis. The solid line is the median and the dashed line
the 95th percentile, both taken in eight temperature bins. Both lines are highest at the hot end,
where the median is about twice that of the colder corners, and the 95th percentile is also raised
at the cold end. Temperature is the corner variable that the error follows most. Its rank
correlation with |error| is 0.20 for the gates and 0.26 for the adder, and for VDD or either vth0 it
is below 0.15 in magnitude.

The errors grow with temperature because the share of the OFF current grows. The gate current of the
ON transistors is nearly all of the leakage at cold corners and 83 % of it (median) above 92 °C. The
OFF currents are predicted less well than the ON gate currents (@tab-transistor-errors), so they
weigh more at hot corners.

#figure(
  placement: auto,
  image("/report/analysis/corner_error.pdf", width: 100%),
  caption: [Where the errors are largest. On the left is the gate error of every simulation against temperature, with lines for the median and the 95th percentile in eight temperature bins. On the right, each point is one of the 200 corners, with the median signed error of the gates on the horizontal axis and of the adder on the vertical axis. The three corners with the largest adder errors are named.],
) <fig-corner>

The adder shows where the large errors are. Of its 256 largest errors (the worst 1 %), 216 come from
three corners, the three circled points in the right panel of @fig-corner. Each point of that panel
is one corner, placed by the median signed error of the gates (horizontal) and of the adder
(vertical). The points lie along the diagonal, with a rank correlation of 0.83, so a corner that
over-predicts the gates over-predicts the adder by about as much. At one corner every gate sees the
same VDD, temperature and process, so the errors of the gates are correlated. Summing 42 gates
averages out the differences between gate cases but not the error a corner puts on all of them.

Each of the three corners has several variables near the ends of their ranges at once. Corner 133 is
at -28 °C with VDD at 0.94 V and the NMOS vth0 9.9 % above nominal, corner 1 at 107 °C with VDD at
0.90 V, and corner 76 at 80 °C with the NMOS vth0 9.9 % above and the PMOS |vth0| 7.5 % below
nominal. The models are least accurate near the ends of the sampled ranges, where a uniform draw
leaves the fewest training samples around a point. That is the same effect as the 1 Lmin width.

We also traced the largest single errors. For each of the 20 largest gate errors we split the error exactly into the contributions of the predicted currents, since the Kirchhoff sum is linear in them. The
same three causes come out on the 200 corners of Part B and on the held-out ones.
- The gate current of an ON NMOS with its input at VDD (INV 1, NOR2 10 and 11), over-predicted by 5 to
  16 %. VDD is 0.91 to 0.96 V at all of these corners, and all but one are cold, -31 to -7 °C.
- The source current of the OFF PMOS stack of NOR2 11 and OR2 11 at 118 to 125 °C, under-predicted by
  5 to 16 %. The largest held-out error, 16.2 %, is this case.
- The drain current of an OFF NMOS, under-predicted by 4 to 9 %. Most of these errors are INV 0 at
  35 to 123 °C, and the same current causes the NOR2 00 and OR2 11 errors on the list.

= Limitations and improvements

The accuracy is limited by the transistor models near the ends of the sampled ranges, and in a gate the
error of each model shows up in proportion to its share of the leakage.

- Hot corners. The OFF drain current is the least accurate current that the gate leakage uses (@tab-transistor-errors), and the gate error is largest at the hot end of the temperature range (@fig-corner). In a technology with less gate
  tunnelling the OFF current would set the gate error.
- Narrow ON NMOS. Its gate current at 1 Lmin is still over-predicted by about 1.2 %, and it is behind the largest cold-corner errors.
- ON drain and source currents. The final models give them the wrong sign in 10 to 12 % of the
  targeted test samples. That is harmless with the Kirchhoff sum, but a user who sums the
  terminal currents directly gets #fixed(gate-terminal.at(-1).median_error_percent, 1) % median gate
  error. The Kirchhoff sum also needs the topology of the gate and which transistors are ON, so it is
  part of the method and not of the models.
- One technology and one circuit. The test covers PTM 22nm LP with these five gates and one adder, and the SPICE
  reference includes our choice of diffusion geometry. Without diffusion areas, which the course
  netlist does not set, the leakage is up to 27.6 % lower at hot corners.
- Fitting. The learning curve has not levelled off at 10,500 samples per case (@fig-learning), and
  each model type was fitted with one set of hyperparameters.

Two changes would address these limits. First, more training samples where the errors are, at the small widths and near
the hot end of the temperature range. Training on the gate-bias set as it stands (change A3) failed its condition, so the samples would have to be added without giving up the OFF drain current across the
rails. Second, a separate model for the OFF channel current, as change A2 did for the ON gate current, so
that the two do not trade against each other. Any further evaluation needs new corners, since we have now seen the results on the 400 of @sec-blind.

#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])

= Technology and tools

#figure(
  placement: none,
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([Item], [Value]),
    [Model file], [PTM 22nm LP, BSIM4 (level 54, version 4.0)],
    [Nominal VDD], [1 V, sampled from 0.9 to 1.1 V. The model file states 0.95 V],
    [Channel length and widths], [L = Lmin = 22 nm, W from 1 to 8 Lmin],
    [Test circuit], [3-bit carry-lookahead adder (CLA), 42 gates and 204 transistors],
    [Simulator], [ngspice-45, operating point (`op`) analysis],
    [Netlist options], [`.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` in every netlist],
    [Software], [Python with NumPy, pandas, scikit-learn and LightGBM, PyTorch for the MLP, from a Nix
      flake (`flake.nix`)],
    [Hardware], [NVIDIA GeForce RTX 4070 Laptop GPU with 20 CPU cores],
  ),
  caption: [Technology, test circuit and tools.],
) <tab-setup>

= Supporting checks <sec-checks>

== SPICE reference <sec-reference-checks>

*Reported currents.* The reference uses the terminal currents ngspice reports for each transistor.
The currents of the voltage sources carry a rounding error near 1e-16 A, so we kept them only as a
cross-check. The two agree within 1.0e-4 relative and 6.7e-16 A absolute over all 3,600 gate
simulations.

*Drain current.* In ngspice `@m[id]` is only the channel current. The drain terminal current also
takes GIDL, the drain junction, impact ionisation and the gate-to-drain tunnelling, so we build it
from the terms that the ngspice-45 source code adds at the drain node. We checked it against the
source currents of single transistors in 72 netlists (four cases, six pairs of drain and source
voltage, three temperatures). In all 64 where the currents are at least 1e-15 A the two agree
exactly, while `@m[id]` alone is off by about 100 % (median).

*Gate resistor.* The gate and adder netlists set `rgatemod = 0` in the transistor models. With
`rgatemod = 1` the gate resistor is 0.13 Ω. A gate driven by another gate's output near VDD then
carries about 2e-15 A of rounding noise against a current tolerance of 1e-18 A (`abstol`), and some
AND2 operating points never converge. Of the 3,600 netlists 61 failed, all AND2, and two more
converged to a point where the currents at the supply do not sum to zero. With `rgatemod = 0` all
3,600 converged. On the other 3,537 the two settings differ by at most 0.51 %, and by at most 5e-5
in 99 % of them.

== Scoring and composition code <sec-code-checks>

*Constant prediction.* As a check of the scoring code of @tab-comparison we scored a constant
prediction, the training median of each terminal. It gets an R² below 0 and a median error of 98 %
or more.

*SPICE currents in place of the models.* With each transistor's own SPICE currents in place of the
models, the code that builds the gate leakage must give back the SPICE leakage. It does, within
#pct-wide(gate-oracle.at(-1).max_error_percent) % with the Kirchhoff sum and
#pct-wide(gate-oracle-terminal.at(-1).max_error_percent) % with the plain sum at worst.

== Adder netlist <sec-adder-checks>

We ran two checks on the adder netlist. At the nominal corner the outputs give A + B + Cin for all
128 vectors. Then at every corner all 42 gate output nets of all 25,600 simulations, 1,075,200
checks, match the logic values our code computes to decide which inputs each gate sees. The nets sit
within 2.3e-6 VDD of their rail (median) and 1.5e-4 VDD at worst.

== Selecting the changes <sec-protocol>

The gate-bias score varies between two trainings of the same method. To measure this variation we
trained the baseline 8 times with different random seeds (first row of @tab-arms). Each change has
its own condition for dropping it. Where a condition asks whether a metric improves, the metric
counts as improved only if its mean over 3 trainings with different seeds is lower than the baseline
mean by more than 2 standard deviations. The five changes, their conditions and this rule were
written down before any of these runs. @tab-arms reports
every run, with the condition for dropping each change and the value we measured for it.

#let arm-info = (
  A1: ([ON Ig and Ib per effective width W - 2 wint], [Ratio of the ON Ig error at 1 Lmin to that at
    wider devices stays at 1.5 or more], [1.69]),
  A2: ([Second MLP for ON Ig and Ib], [ON Ig error improves less than 1.5 times], [3.48 times]),
  A3: ([Train on the gate-bias set as well], [OFF Id error with drain and source on opposite rails
    above 5 %], [9.0 %]),
  A4: ([OFF Ig at or below 1e-26 A out of the loss], [OFF Id error on opposite rails does not improve],
    [11.9 to 5.0 %]),
  A5: ([Ensemble of 5 seeds, mean of log#sub[10]|I|], [Score improves less than 5 %], [44 %]),
  A2A4: ([A2 and A4 together], [], []),
  A2A4A5: ([A2, A4 and A5 together], [], []),
)
#figure(
  placement: auto,
  small-table(table(
    columns: (auto, 1fr, auto, 1.25fr, auto, auto),
    align: (left, left, right, left, left, left),
    inset: 4pt,
    table.header([Change], [What it does], [Gate-bias score~#good-down], [Dropped if], [Measured], [Result]),
    [Baseline], [MLP, log target, uniform and targeted sets],
    [#fixed(noise.mean, 4) ± #fixed(noise.std, 4)], [], [], [],
    ..arm-info.keys().map(label => {
      let row = arm(label)
      let cells = (
        [#label], arm-info.at(label).at(0),
        [#fixed(row.gatebias_score_mean, 4) ± #fixed(row.gatebias_score_std, 4)],
        arm-info.at(label).at(1), arm-info.at(label).at(2),
        if row.killed == "True" [dropped] else if row.killed == "False" [kept] else if label == "A2A4A5" [final] else [],
      )
      if label == "A2A4A5" { cells.map(body => table.cell(fill: accent-light, body)) } else { cells }
    }).flatten(),
  )),
  caption: [The five changes and their two combinations. Each score is the mean ± standard deviation of the gate-bias score on the Part A validation samples over 3 trainings, or 8 for the baseline. _Dropped if_ is the condition we wrote down before the runs, and _Measured_ is what the runs gave. The final method is shaded.],
) <tab-arms>

*A1, gate current per effective width (dropped).* The ON gate current is less accurate for narrow
transistors than for wide ones (@sec-errors). In the model file and in a sweep of W alone that
current scales with the effective width W - 2 wint. So A1 fits the current divided by this width, to
take the width out of what the network has to learn. This did not improve the narrow transistors.
Their error stayed above the limit we had set for it, and the score stayed inside the band.

*A3, training on the gate-bias set (dropped).* The gate-bias samples have the voltages a transistor
has in a gate, so training on them is the direct fix. It lowered the score. But the OFF drain
current with drain and source on opposite rails stayed less accurate than the limit we had set, so
we dropped it.

*The final model.* Whether the ensemble is part of the final model was fixed by a rule written
before A2 and A4 with or without it were run on any circuit. The rule takes the ensemble only if it
beats A2 and A4 by more than 2 baseline standard deviations. The ensemble met this rule narrowly,
with an improvement of 0.0097 against a threshold of 0.0093.

== Held-out corners <sec-statistics>

The 95 % intervals of @tab-blind come from a bootstrap over corners, not single simulations, with
10,000 draws and seed 463, since the simulations of one corner share its errors. We looked at the
results once and changed nothing afterwards.

@tab-transistor-r2 checks the transistor currents themselves at the held-out corners. It gives R² of
log#sub[10]|I| for each case and terminal, between the currents the gate leakage takes from the
models and the currents SPICE recorded in the same simulations. Every current is at 0.99 or above
except the OFF gate currents (shaded in orange). Of those, 69 % (NMOS) and 67 % (PMOS) are at or
below 1e-26 A, more than 14 decades below the smallest gate leakage of Part B, and R² counts only
the rest. Their fit does not affect a gate. Over all currents R² is
#fixed(load("/results/stage4/transistor_r2_frozen.csv").at(-1).r2_log10, 4), higher than the 0.9771
of the baseline models and within 0.001 of the value on the 200 corners of Part B.

#let r2-blind = load("/results/stage4/transistor_r2_frozen.csv")
#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header([Case], [Id], [Ig], [Is], [Ib]),
    ..case-names.keys().map(case-name => (
      case-names.at(case-name),
      ..("id", "ig", "is", "ib").map(current => {
        let found = r2-blind.find(row => row.case == case-name and row.current == current)
        if found == none [Kirchhoff] else if case-name.ends-with("_off") and current == "ig" {
          table.cell(fill: warn-light, fixed(found.r2_log10, 4))
        } else { fixed(found.r2_log10, 4) }
      }),
    )).flatten(),
  ),
  caption: [R² (#good-up) of log#sub[10]|I| for the currents of the final models in the gates at the held-out corners, counting currents above 1e-26 A. The ON drain and source currents come from Kirchhoff's law and not from a model.],
) <tab-transistor-r2>

= Deliverables

The repository is #link("https://github.com/MostlyKIGuess/trickle-up")[github.com/MostlyKIGuess/trickle-up].
Paths are relative to its root, and the README gives the commands that regenerate each
file (@tab-deliverables, @tab-sources). Most numbers quoted only in the text come from the CSV files
in `results/` and `report/analysis/`. The 2.07 % of the baseline comes from the predictions of the first version, and a
few counts come from one-off checks of the sample sets.

#figure(
  placement: none,
  small-table(table(
    columns: (auto, 1fr, 1.35fr, auto),
    align: (right, left, left, left),
    inset: 4pt,
    table.header([\#], [Deliverable], [File], [Report]),
    [1], [Datasets, four CSV files of 10,000 samples and 16 columns],
    [`datasets/fet_nmos_on.csv`, `fet_nmos_off.csv`, `fet_pmos_on.csv`, `fet_pmos_off.csv`],
    [@sec-sampling],
    [2], [Example netlists], [`spice/fet_<case>.cir` (4), `spice/inv.cir`, `nand2.cir`, `nor2.cir`,
      `and2.cir`, `or2.cir`, `spice/cla3.cir`], [@sec-netlists, @sec-reference],
    [3], [Data check], [`results/kirchhoff.csv`, `results/figures/trend_*.pdf`, `results/trend_sweeps.csv`],
    [@tab-kirchhoff, @fig-trends],
    [4], [Model comparison, 4 types × 4 cases], [`results/comparison/model_comparison.csv`,
      `report/analysis/model_summary.csv`], [@tab-comparison],
    [5], [Chosen models, test errors, parity plots, model files], [`results/chosen_models.csv`,
      `results/figures/parity_<case>.pdf`, `models/fet_<case>.joblib` (4)], [@tab-chosen, @fig-parity],
    [6], [Gate results table, 18 cases, and the record of every simulation (leakages and node
      voltages)], [`results/gate_results_ml.csv`, `report/analysis/gate_records.csv`], [@tab-gates],
    [7], [Gate parity plot, log axes], [`results/figures/gate_parity_ml.pdf`], [@fig-gate-parity],
    [8], [Test-circuit results and parity plot], [`results/cla_results_ml.csv`,
      `results/figures/cla_parity_ml.pdf`], [@tab-cla, @fig-cla],
    [9], [Code, run in order with fixed seeds], [`scripts/01_*.py` to `scripts/13_*.py`,
      `scripts/run_all.sh`, `flake.nix`], [README],
    [10], [Typed report], [`report/report.typ`, `report/report.pdf`], [this document],
  )),
  caption: [Deliverables of the assignment and where to find them.],
) <tab-deliverables>

#figure(
  placement: none,
  small-table(table(
    columns: (auto, 1fr),
    align: (left, left),
    inset: 4pt,
    table.header([Table or figure], [File]),
    [@tab-kirchhoff], [`results/kirchhoff.csv`],
    [@fig-vds, @fig-currents], [`report/analysis/sample_vds.pdf`, `current_ranges.pdf`, `on_vds.csv`],
    [@fig-trends], [`results/figures/trend_sweep.pdf`, `report/analysis/trend_sweeps.csv`],
    [@fig-learning], [`report/analysis/learning_curve.pdf`, `results/comparison/learning_curve.csv`],
    [@fig-arms, @tab-arms], [`results/arms/arm_report.csv`, `noise_floor.csv`, `results/arms/runs/`],
    [@tab-cla], [`results/cla_results_*.csv`, `results/cla_composition_control.csv`],
    [Time of SPICE and of our method (@sec-adder)], [`results/timing.csv`],
    [@tab-blind, @tab-transistor-r2], [`results/stage4/stage4_summary.csv`, `transistor_r2_frozen.csv`],
    [@fig-attribution, @tab-transistor-errors], [`report/analysis/attribution.csv`, `transistor_errors.csv`],
    [@fig-corner], [`report/analysis/corner_errors.csv`, `per_corner.csv`],
    [Checks of the SPICE reference], [`results/diffusion_comparison.csv`, `rgatemod_comparison.csv`,
      `drain_current_check.csv`, `gate_device_vs_source.csv`, `cla_logic_check.csv`],
  )),
  caption: [Files behind the other tables and figures.],
) <tab-sources>

#heading(numbering: none)[Contributions]

Krish Pandya and Vignesh Vembar sat together and did the work, writing the code and this report with
the help of Claude, an AI assistant. Aditya Peketi and Sudhanva Joshi verified the results at every
step. So the team worked like an actor and a critic. One pair proposed and ran each step, and the
other pair checked it before the next one started. All four of us contributed equally. The code,
data and models are at
#link("https://github.com/MostlyKIGuess/trickle-up")[github.com/MostlyKIGuess/trickle-up].
