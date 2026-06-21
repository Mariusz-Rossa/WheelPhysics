# WheelPhysics

**Wheel Algebra applied to singularities in theoretical physics.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20305458.svg)](https://doi.org/10.5281/zenodo.20305458)
![Status](https://img.shields.io/badge/status-experimental%20research-orange)
![Python](https://img.shields.io/badge/python-3.11+-blue)

```bash
git clone https://github.com/Mariusz-Rossa/WheelPhysics.git
cd WheelPhysics
pip install sympy numpy
python main.py --quick
```

---

## What is this?

Most computer algebra systems treat division by zero as an error - they simplify
expressions aggressively and abort at singular points, sometimes losing structural
information in the process.

**Wheel Algebra** (Carlström 2004) is an algebraic extension of commutative rings in
which division by zero is a first-class operation, producing a well-defined absorbing
element **⊥** ("bottom") instead of an error. This project explores whether
singularity-aware symbolic evaluation can:

- preserve denominator structure at singular points
- distinguish removable singularities from genuine poles
- track how ⊥ propagates through algebraic expressions
- support systematic analysis of singular points in physical equations

Applied to 46 canonical equations of theoretical physics across six domains.

---

## Non-goals

This project does **not** claim to:

- solve quantum gravity
- remove renormalization from QFT
- replace standard distribution theory or regularisation methods
- redefine physical observables

WheelPhysics is a symbolic-algebra research framework for studying the structure and
classification of singularities in physical equations.

---

## Key results

| Result | Details |
|--------|---------|
| **Kretschmann invariant** | K(r=r_s) = 12/r_s⁴ (finite), K(r=0) = ⊥ - algebra distinguishes coordinate artefact from physical singularity |
| **On-shell singularities map to ⊥** | All tested Feynman propagators produce ⊥ at the on-shell condition; off-shell (virtual particles) remain finite and computable |
| **Resonance ↔ on-shell isomorphism** | res(1/(ω²−ω₀²), ω=ω₀) = 1/(2ω₀) - algebraically identical to res(1/(p²−m²), p=m) = 1/(2m) |
| **Logarithmic poles (QCD)** | Gluon propagator 1/(k²·(1+αs·log(k²/μ²))) has two distinct log poles: Landau pole (order=1, res=1/αs) and IR pole at k²=0 (no algebraic order, lim k²ⁿ·f=0 ∀n) |
| **Removable singularities** | sinc(0): Wheel=⊥, analytic layer recovers lim=1 |
| **46 equations, 60 singular points** | GR, cosmology, QFT, classical mechanics, thermodynamics, pure math |
| **All current verification tests pass** | Automated consistency suite on all expressions |

---

## Example

```python
from sympy import symbols, sin
from core.wheel_algebra import WheelAlgebra, W
from core.wheel_calculus import analyse_singularity

x, r, r_s = symbols('x r r_s')
wa = WheelAlgebra()

# Division by zero → ⊥
expr = W(1 / r)
result = wa.evaluate_at(expr, {r: 0})
print(result)   # ⊥

# Removable singularity: Wheel returns ⊥, calculus recovers the limit
expr2 = sin(x) / x
analysis = analyse_singularity(expr2, [(x, 0)], name="sinc")
print(analysis)  # RemovableSingularity(limit=1)

# Schwarzschild metric: coordinate pole at r=r_s
expr3 = W(1 / (1 - r_s / r))
result3 = wa.evaluate_at(expr3, {r: r_s})
print(result3)   # ⊥  (PoleSingularity, order=1, residue=r_s)
```

---

## Singularity classification

The framework stratifies singularities into five categories (v1.1):

```
Wheel result at x₀:

  Finite(v)    →  regular point
  ⊥  ──┬──→  0/0 form        →  Taylor      →  RemovableSingularity(lim=v)
       ├──→  algebraic pole  →  residue     →  PoleSingularity(order, res, Laurent hint)
       ├──→  logarithmic pole→  sp.residue()→  LogarithmicSingularity(order, res|None)
       └──→  unclassified    →  fallback    →  ⊥ (UNKNOWN)
```

Formal type system: `SingularityType` enum with 12 types:

| Type | Symbol | Description |
|------|--------|-------------|
| `REGULAR` | ✓ | Regular point, Wheel finite |
| `REMOVABLE` | ⊥→v | 0/0 form, limit exists (Taylor) |
| `POLE` | ⊥ | Algebraic pole, general (order unknown) |
| `POLE_SIMPLE` | ⊥ | Order-1 pole, Cauchy residue defined |
| `POLE_HIGHER` | ⊥ | Higher-order pole (order ≥ 2), residue N/A |
| `ESSENTIAL` | ⊥ | Essential singularity (Picard–Weierstrass), e.g. exp(1/z) at z=0 |
| `LOGARITHMIC` | ⊥ | Logarithmic divergence (e.g. QCD Landau pole) |
| `BRANCH_POINT` | ⊥ | Branch point, multi-valued function (e.g. √z, log z at z=0) |
| `COORDINATE` | ⊥* | Coordinate artefact (requires invariant) |
| `PHYSICAL` | ⊥ | Confirmed physical singularity |
| `COMPLEX_POLE` | ⊥? | Pole off the real axis (open research question) |
| `UNKNOWN` | ? | Fallback when classification fails |

**Design principle:** `wheel_algebra` (axiomatic) and `wheel_calculus` (analytic) are
deliberately separate modules. Wheel Algebra ≠ limit theory.

---

## Architecture

```
core/
  wheel_number.py       WheelNumber type: values, ⊥, arithmetic (11/11 axioms)
  wheel_algebra.py      Wheel Algebra rules, evaluate_at, rewriting
  sympy_extension.py    SymPy integration: wheel_subs, singularity_map
  wheel_calculus.py     Analytic extension: classification, residues, Laurent hints

scanner/
  singularity_finder.py    Expression scanner: POLE, ZERO_OVER_ZERO, LOGARITHMIC
  translator.py            Classical algebra → Wheel pipeline
  consistency_checker.py   Verification suite

physics/
  equations_db.py          46 equations across 6 domains
  general_relativity.py    Schwarzschild, Kerr, Reissner-Nordström, Friedmann
  quantum.py               Feynman propagators, UV divergences, Dirac, on-shell
  results_log.py           Generated logs (not hardcoded)

results/
  wheel_results.json       46 entries: ⊥=38, finite=4, mixed=4
  calculus_results.json    14 entries with pole order + residue
```

---

## Usage

```bash
python main.py                    # full analysis
python main.py --quick            # project summary
python main.py --module gr        # general relativity only
python main.py --module qft       # quantum field theory only
python main.py --module calculus  # singularity classification + JSON output
python main.py --module viz       # ASCII plots with ⊥ markers
python main.py --regen-log        # recompute wheel_results.json
python main.py --calculus-log     # show calculus_results.json
python main.py --db               # equation catalogue
```

---

## Contributing

This is a solo-maintained research project, but issues, bug reports, and
discussion are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
guidelines, and [`CHANGELOG.md`](CHANGELOG.md) for release history.

Open research directions are tracked as GitHub issues - see in particular
the open question on complex poles (`COMPLEX_POLE`, off the real axis).

---

## Preprint

Draft v0.2 available in [`Preprint.md`](Preprint.md) and on Zenodo:
[https://doi.org/10.5281/zenodo.20314153](https://doi.org/10.5281/zenodo.20314153)

**Title:** *A Singularity-Aware Symbolic Algebra Framework for Physical Equations*

---

## Background

Wheel Algebra was introduced by Carlström (2004) as an algebraic extension of
commutative rings admitting division by zero. This project uses Carlström's formulation
(single absorbing element ⊥) rather than Meyenburg's (2025) semi-ring extension
(directed infinities ±∞). The directional structure of poles is recovered by the
analytic layer (`wheel_calculus`) rather than built into the base algebra.

---

## Citation

If you use this project in academic work, please cite the Zenodo release:

```bibtex
@software{wheelphysics_2026,
  author    = {Rossa Mariusz},
  title     = {WheelPhysics: A Singularity-Aware Symbolic Algebra Framework},
  month     = may,
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20305458},
  url       = {https://github.com/Mariusz-Rossa/WheelPhysics}
}
```

---

## License

MIT - see [`LICENSE`](LICENSE).

---

*Independent research project. All results reproducible: `python main.py`.*