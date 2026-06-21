# Contributing to WheelPhysics

Thanks for your interest in WheelPhysics. This is an independent research
project exploring whether Wheel Algebra (Carlström 2004) can provide a formal
apparatus for analyzing singularities in theoretical physics. It's
maintained solo, but issues, questions, and contributions are welcome.

## Ways to contribute

- **Bug reports.** If something in `wheel_algebra.py`, `wheel_calculus.py`, or
  any physics module gives an incorrect or inconsistent result, please open an
  issue. Include the expression, the singular point, and the expected vs.
  actual classification.
- **New equations.** `equations_db.py` catalogues equations across six physics
  domains. Suggestions for additional equations with known, citable
  singularities (with a reference) are welcome - open an issue or a PR.
- **New singularity types.** The `SingularityType` system is meant to grow
  (e.g. `COMPLEX_POLE` is currently open). If you have a
  mathematically sound approach for a type not yet handled, open an issue to
  discuss before submitting a PR.
- **Documentation.** Clarifications, typo fixes, and improved examples in the
  README, `Preprint.md`, or docstrings are always appreciated.
- **Discussion.** Open an issue even for "is this even a sensible question"
  type discussions - the open questions list in the README is a fair
  reflection of where the project is genuinely uncertain.

## Before opening a PR

1. **Open an issue first** for anything beyond a trivial fix (typos, obvious
   bugs). This project makes specific architectural choices (e.g. Carlström's
   single `⊥` over Meyenburg's directed infinities - see README "Background")
   and it's better to align on approach before writing code.
2. **Keep changes scoped.** One topic per PR, mirroring the commit convention
   below.
3. **Run the consistency suite** before submitting:
   ```bash
   python main.py --module consistency
   python main.py --module calculus
   ```
   New code should not reduce the pass rate reported by `ConsistencyChecker`.

## Commit conventions

This project favors small, descriptive commits over large batched ones:

- One topic per commit.
- Descriptions in English, specific and in the imperative mood:
  `add Coulomb singularity to equations_db`, not `various fixes`.
- Reference issue numbers where relevant.

## Code style

- Python 3.11+, SymPy for all symbolic work.
- Keep the separation between `wheel_algebra.py` (axiomatic: always `⊥` at
  undefined forms) and `wheel_calculus.py` (analytic: classification,
  residues, limits) intentional - don't fold limit-taking logic into the
  algebra layer.
- New singularity detection logic should preserve denominator factor
  structure (see `LogarithmicSingularity` for the `sp.denom()` +
  `Mul.make_args()` pattern) rather than relying on aggressive
  simplification, which can erase the structure being analyzed.

## Scope

This project does **not** aim to solve quantum gravity, remove renormalization
from QFT, or replace standard distribution/regularization theory (see
README "Non-goals"). Contributions that fit within "symbolic-algebra
framework for classifying singularities" are in scope; contributions that
require taking a side on open physics questions the project is deliberately
agnostic about are better suited to a research issue/discussion first.

## Questions

Open an issue, or reach out via the contact details on the
[Zenodo record](https://doi.org/10.5281/zenodo.20305458) / ORCID
([0009-0006-1060-2883](https://orcid.org/0009-0006-1060-2883)).