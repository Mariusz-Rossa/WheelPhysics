# Changelog

All notable changes to WheelPhysics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
on a best-effort basis (research code - APIs may still shift between minor versions).

## [Unreleased]

### Planned
- Complex pole support (`COMPLEX_POLE`) - extend Wheel evaluation to ℂ 
- Classical electrodynamics: Coulomb singularity, Liénard–Wiechert potentials 
- Schrödinger equation with singular potentials (1/r, delta) 
- Extend `equations_db` to 60+ equations 
- AI usage disclosure section in `paper.md` 
- Preprint v0.3: logarithmic and complex pole results 

## [1.1.0] - 2026-05-26

### Added
- `LogarithmicSingularity` result type in `wheel_calculus.py` - automatic detection
  of logarithmic poles via `sp.denom()` + `Mul.make_args()`, preserving denominator
  factor structure instead of collapsing it under simplification.
- Residue computation for logarithmic poles via `sp.residue()`, with a `sp.limit()`
  fallback for indeterminate forms (e.g. `0·log(0)`) where `.subs()` returns `nan`.
- Five-fold singularity classification (`WheelFinite`, `RemovableSingularity`,
  `PoleSingularity`, `LogarithmicSingularity`, fallback `⊥`), up from the
  four-fold split in v1.0.

### Research findings
- QCD gluon propagator `1/(k²·(1+αs·log(k²/μ²)))` exhibits two qualitatively
  distinct logarithmic poles:
  - **Landau pole** at `k² = μ²·exp(-1/αs)`: algebraic order = 1, residue = `1/αs`.
  - **IR pole** at `k² = 0`: no algebraic order exists (`lim kⁿ·f = 0` for all `n`) —
    a divergence strictly stronger than any single algebraic pole order.
- Closed issues (`Logarithmic poles: Poincaré asymptotic series for QCD Landau
  pole`) - `sp.residue()` + `sp.limit()` proved sufficient for classification;
  a full Poincaré asymptotic series was not required.

### Fixed
- Import paths in `wheel_calculus.py` corrected for the flat repository layout
  (no `core/` subpackage).

## [1.0.0] - 2026-05-20

### Added
- Initial public release.
- `WheelNumber` class implementing Carlström (2004) Wheel Algebra: `+`, `-`, `*`,
  `/`, `/0 = ⊥` (`BOTTOM`), with 11/11 axiom tests passing.
- `wheel_algebra.py`: rewriting rules, `evaluate_at`, `wheel_limit`, and the core
  `_has_division_by_zero_at()` recursive AST check for denominators.
- `sympy_extension.py`: `wheel_subs`, `WheelFunction`, `singularity_map`,
  `expr_to_wheel`, `wheel_series_around`.
- `wheel_calculus.py` (v1.0): four-fold `SingularityType` classification —
  `REMOVABLE`, `POLE_SIMPLE`, `POLE_HIGHER`, fallback `⊥` - with residue and
  Laurent-hint computation for algebraic poles.
- `singularity_finder.py`: expression scanner for `POLE`, `ZERO_OVER_ZERO`,
  `LOGARITHMIC` patterns.
- `translator.py`: classical algebra → Wheel pipeline.
- `consistency_checker.py`: verification suite, 100% pass on 5 expressions,
  10/10 classification tests.
- `equations_db.py`: catalogue of 46 equations across 6 physics domains,
  60 singular points total.
- `general_relativity.py`: Schwarzschild, Kerr, Reissner–Nordström, Friedmann,
  Christoffel symbols, Kretschmann invariant.
- `quantum.py`: Feynman propagators, UV divergences, Dirac equation, on-shell
  conditions.
- `results_log.py`: results generated from live analysis (not hardcoded),
  written to `wheel_results.json` and `calculus_results.json`.
- `comparator.py`: ASCII visual comparisons with `⊥` markers.
- `main.py`: CLI entry point (`--quick`, `--module`, `--regen-log`,
  `--calculus-log`, `--db`).
- `paper.md` / `paper.bib`: JOSS submission draft.
- `Preprint.md` / `Preprint.pdf`: preprint v0.2, published to Zenodo.
- GitHub Action `draft-pdf.yml`: compiles `paper.md` to PDF on every push.
- MIT license (code), CC BY 4.0 (preprint).

### Research findings
- Wheel Algebra distinguishes coordinate artefacts from physical singularities:
  Kretschmann invariant is finite at the Schwarzschild radius (`K(r_s) = 12/r_s⁴`)
  but `⊥` at `r = 0`.
- On-shell states map to `⊥` across all tested Feynman propagators.
- Classical resonance `1/(ω² − ω₀²)` is algebraically isomorphic to the QFT
  propagator pole structure: `res(1/(ω²−ω₀²), ω=ω₀) = 1/(2ω₀)`, matching
  `res(1/(p²−m²), p=m) = 1/(2m)`.
- `sinc(x)` at `x = 0` is a documented counterexample showing Wheel Algebra is a
  pointwise algebra, not a limit theory.
- `Γ(ε) = ⊥` and `ζ(1) = ⊥` support the hypothesis that dimensional and zeta
  regularization act as classical substitutes for Wheel.

### Known issues
- Fermionic propagator at `p = −mₑ` returns `classical = "nan"` (0/0 form)
  instead of `"zoo"` (±∞); Wheel correctly returns `⊥` regardless. Flagged as a
  known limitation pending further investigation (issue #9).
- UV divergences live in `∫d⁴k` integrals; Wheel currently operates pointwise
  and does not address integral divergences (issue #10).