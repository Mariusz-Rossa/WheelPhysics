# A Singularity-Aware Symbolic Algebra Framework for Physical Equations

**Draft v0.1 — WheelPhysics Project**
*Last updated: 2026-05-19*
*Status: SZKIELET — sekcje do rozwinięcia oznaczone [TODO]*

---

## Abstract

*[TODO — napisać po dopięciu preprintu. Sugerowana struktura: problem (osobliwości
w fizyce teoretycznej) → metoda (Wheel Algebra Carlströma + warstwa calculus) →
kluczowe wyniki (46 równań, czwórpodział, residue analysis, izomorfizm rezonans↔on-shell)
→ dostępność kodu.]*

Singular points in theoretical physics — black hole singularities, the Big Bang at t=0,
infrared and ultraviolet divergences in quantum field theory — are routinely treated as
computational breakdowns rather than mathematical objects. We present a symbolic algebra
framework based on Wheel Algebra (Carlström 2004) in which division by zero produces a
well-defined absorbing element ⊥ ("bottom") instead of an error. Applied to 46 equations
across six physical domains (general relativity, cosmology, quantum field theory, classical
mechanics, thermodynamics, and pure mathematics), the framework correctly identifies
singular points, classifies them into four structural categories, and — for pole
singularities — computes residues via the Cauchy theorem. We demonstrate a formal
isomorphism between classical resonance and the QFT on-shell condition, and show that
the Kretschmann scalar allows the framework to distinguish coordinate artefacts from
genuine physical singularities. The complete implementation is available as verifiable
Python code.

---

## 1. Introduction

### 1.1 The problem: singularities as computational failures

Physical theories break down at singular points in two distinct ways. Some singularities
are artefacts of the coordinate system — the Schwarzschild horizon at r = r_s is a
well-known example, regular in Kruskal coordinates but singular in Schwarzschild
coordinates. Others are genuine physical singularities: the curvature invariant
K = 12r_s²/r⁶ diverges at r = 0 regardless of coordinate choice.

Classical analysis handles this distinction by external argument (change of coordinates,
check invariants). The algebra itself does not encode it.

A second class of problems arises in quantum field theory. The Feynman prescription
(i ε in the propagator denominator) is an explicit workaround: the physical pole
1/(p² − m²) is displaced off the real axis to avoid division by zero during momentum
integration. The prescription works, but its justification is procedural rather than
algebraic.

In both cases, division by zero appears as an obstacle to be circumvented. We ask a
different question: what if division by zero were admitted as a first-class algebraic
operation?

### 1.2 Wheel Algebra

*[TODO — skrócone wprowadzenie do Wheel Algebra Carlströma. 1-2 strony.
Kluczowe aksjomaty: /0=⊥, ⊥ absorbujące, 0·⊥=⊥. Odróżnienie od IEEE 754 NaN.
Odróżnienie od Meyenburga (2023) — świadoma decyzja architektoniczna.]*

Wheel Algebra, introduced by Carlström (2004), extends a commutative ring with two
operations: inversion `/x` (not reciprocal `1/x`) and an absorbing element ⊥ ("bottom")
that captures all undefined forms. The axioms include:

- `/0 = ⊥`
- `⊥ + x = ⊥` for all x (absorption under addition)
- `⊥ · x = ⊥` for all x (absorption under multiplication)
- `0/0 = ⊥`, `1/0 = ⊥`, `∞/∞ = ⊥` — all indeterminate forms map to the same element

Crucially, ⊥ is not an error — it is an element of the algebra. Expressions containing ⊥
remain algebraically manipulable.

We deliberately use Carlström's formulation rather than Meyenburg's (2023) semi-ring
extension, which introduces distinct `+∞` and `−∞` elements. The directional structure of
poles (which way does the function blow up?) is recovered by the analytic layer
(`wheel_calculus`) rather than built into the base algebra. This separation is an
intentional architectural decision; we discuss it in Section 2.

### 1.3 Scope and contributions

This paper makes the following contributions:

1. **A working implementation** of Wheel Algebra as a SymPy extension, with 11/11 ring
   axioms passing automated tests.

2. **An analytic extension layer** (`wheel_calculus`) that stratifies singularities into
   four categories: regular points, removable singularities, poles (with residue and
   Laurent hint), and irreducible ⊥.

3. **A formal singularity type system** (`SingularityType` enum, 12 types) distinguishing
   coordinate artefacts, physical singularities, logarithmic divergences, and complex poles.

4. **Application to 46 equations** across GR, cosmology, QFT, classical mechanics,
   thermodynamics, and pure mathematics — with 60 known singular points catalogued.

5. **A formal isomorphism** between classical resonance and QFT on-shell condition,
   documented via residue analysis.

---

## 2. Architecture

### 2.1 Design principles

The framework is organised around a clean separation:

```
wheel_number.py     — WheelNumber type: values, ⊥, arithmetic
wheel_algebra.py    — Wheel Algebra rules, evaluate_at, rewriting
sympy_extension.py  — SymPy integration: wheel_subs, singularity_map
wheel_calculus.py   — Analytic extension: classification, residues, Laurent
```

**Wheel Algebra ≠ limit theory.** This separation is fundamental. `wheel_algebra.py`
implements the axiomatic algebra: at a singular point it returns ⊥, unconditionally.
`wheel_calculus.py` is a separate diagnostic layer: when Wheel returns ⊥, it asks *why*
and *what kind*.

### 2.2 The four-way split (wheel_calculus)

```
Wheel result at point x₀:

  Finite(v)              → regular point
  ⊥  ──┬──→ 0/0 form ──→ Taylor expansion ──→  RemovableSingularity(lim=v)
       │                                    └──→ WheelBottom (no structure)
       └──→ true pole ──→ residue analysis ──→  PoleSingularity(order, res, Laurent)
```

For a true pole (numerator ≠ 0, denominator → 0), the analytic layer computes:
- **pole order** n: via `lim (x−x₀)ⁿ · f(x)` for n = 1, 2, ...
- **residue** (order-1 poles): `res = lim (x−x₀) · f(x)` — Cauchy's theorem
- **Laurent hint**: human-readable leading term of the Laurent expansion

### 2.3 The key technical fix: `_has_division_by_zero_at()`

SymPy's `subs()` can silently simplify expressions before substitution, losing
information about division by zero. For example, `1/(1 − r_s/r)` with `r_s → r`
is first simplified by SymPy to `1/(1 − 1) = 1/0`, but if the simplification
proceeds differently the zero in the denominator may be masked.

We solve this by recursively inspecting all denominators in the AST *before*
any substitution. If any denominator evaluates to zero at the target point,
`wheel_algebra.py` returns ⊥ directly, bypassing SymPy's substitution.

### 2.4 Carlström vs Meyenburg: architectural rationale

*[TODO — rozwinąć. Główna teza: Carlström daje jednorodne ⊥ = czysty interfejs
dla warstwy calculus. Meyenburg daje bogatszą algebrę kosztem interfejsu.
Nasze rozwiązanie: jednorodność w algebrze, kierunkowość w calculus.]*

---

## 3. Results

### 3.1 Summary statistics

*[TODO — wstawić po finalnym uruchomieniu `python main.py --regen-log`.]*

| Domain | Equations | Singular points | ⊥ | Finite | Mixed |
|--------|-----------|----------------|---|--------|-------|
| General Relativity | 12 | ~18 | ✓ | — | 4 |
| Cosmology | 5 | 5 | ✓ | — | — |
| QFT | 13 | ~20 | ✓ | 3 | 2 |
| Classical | 4 | 4 | ✓ | — | — |
| Thermodynamics | 6 | 7 | ✓ | 1 | — |
| Mathematics | 6 | 6 | ✓ | — | — |
| **Total** | **46** | **60** | **38** | **4** | **4** |

### 3.2 Kretschmann invariant: distinguishing singularity types

The metric component g_rr of the Schwarzschild solution diverges at both r = r_s
(the horizon) and r = 0 (the geometric centre): both return ⊥. This is correct but
insufficient — g_rr depends on the choice of coordinates.

The Kretschmann scalar K = 12r_s²/r⁶ is a coordinate-independent curvature invariant.
Evaluating both:

```
K(r = r_s)  =  12 / r_s⁴   (finite — the horizon is a coordinate artefact)
K(r = 0)    =  ⊥            (genuine physical singularity)
```

This is our strongest result in GR: Wheel Algebra, applied to a tensorial invariant,
correctly distinguishes the two types of singularity without any external argument.
The distinction is algebraic, not geometric.

### 3.3 Feynman propagators: on-shell = ⊥

Every Feynman propagator in the database becomes ⊥ exactly at the on-shell condition:

| Propagator | Singular point | Wheel result |
|-----------|---------------|--------------|
| `1/(p²−m²)` | p = ±m (on-shell) | ⊥ |
| `1/k²` | k = 0 (massless on-shell) | ⊥ |
| `(p+m)/(p²−m²)` | p = m (electron), p = −m (positron) | ⊥ |
| `1/p` | p = 0 (massless fermion) | ⊥ |
| `1/(s−m²)` | s = m² (Compton) | ⊥ |

Off-shell (virtual particles): all propagators return finite, computable values.

**Hypothesis:** ⊥ on-shell is an algebraic definition of observability in QFT.
Real (observable) particles correspond to algebraically undefined points; virtual
particles are algebraically well-defined. The iε prescription is a classical
workaround for the absence of Wheel Algebra in standard analysis.

*[TODO — rozwinąć argumentację. Porównać z LSZ reduction formalism.]*

### 3.4 Residue analysis: the resonance ↔ on-shell isomorphism

Residue analysis for pole singularities yields:

| Expression | Point | Order | Residue | Physical meaning |
|-----------|-------|-------|---------|-----------------|
| `1/(p²−m²)` | p = m | 1 | `1/(2m)` | On-shell amplitude (QFT) |
| `1/(ω²−ω₀²)` | ω = ω₀ | 1 | `1/(2ω₀)` | Classical resonance |
| `g_rr` at horizon | r = r_s | 1 | `r_s` | Coordinate pole (GR) |
| `1/k²` | k = 0 | 2 | N/A | IR divergence |

The key observation:

```
res(1/(p²−m²),  p = m)   =  1/(2m)
res(1/(ω²−ω₀²), ω = ω₀) =  1/(2ω₀)
```

Both expressions have identical algebraic structure. A classical harmonic oscillator
at resonance and a relativistic particle propagating on-shell are described by the same
residue formula. This isomorphism — between classical resonance and quantum mechanical
on-shell condition — is here documented formally for the first time via residue analysis.

### 3.5 Removable singularities: Wheel ≠ limit theory

Wheel Algebra returns ⊥ at every 0/0 form, including removable singularities. This is
not a deficiency — it is the correct algebraic behaviour. The analytic layer recovers
the limit when it exists:

| Expression | Point | Wheel | Limit |
|-----------|-------|-------|-------|
| `sin(x)/x` | x = 0 | ⊥ | 1 |
| `sin²(x)/x²` | x = 0 | ⊥ | 1 |
| `(1−cos x)/x²` | x = 0 | ⊥ | 1/2 |
| `x/(eˣ−1)` | x = 0 | ⊥ | 1 |

The sinc function, used extensively in signal processing, is a removable singularity.
Wheel + calculus handles it correctly: the algebra flags the undefined form; the
analytic layer recovers the value.

### 3.6 Friedmann equation: continuity through a = 0

*[TODO — rozwinąć. Kluczowe: a<0 matematycznie dozwolone, symetria a→−a,
hipoteza pre-big-bang.]*

### 3.7 Thermodynamic singularities

*[TODO — T=0 → ⊥ (III zasada), entropia BH, dS/dM i propagacja ⊥ przez odwracanie.]*

### 3.8 Open cases

**Logarithmic poles (QCD):** The gluon propagator with one-loop correction contains
a logarithmic divergence (Landau pole). Wheel correctly returns ⊥; Taylor expansion
fails for logarithms. The appropriate analytic tool is an asymptotic (Poincaré) series.
This is classified as `LOGARITHMIC` in `SingularityType` and remains open.

**Complex poles:** The damped harmonic oscillator Green's function has poles at
ω = ±√(ω₀² − γ²/4) ± iγ/2, which lie off the real axis for γ > 0. Wheel, operating
over ℝ, does not encounter them at real substitution points. Classified as `COMPLEX_POLE`;
extension to ℂ is open.

---

## 4. Discussion

### 4.1 What does ⊥ mean physically?

*[TODO — trzy hipotezy: (a) stan nieosiągalny, (b) zakaz algebraiczny,
(c) coś trzeciego — granica stosowalności teorii.
Porównać: p=0 dla bezmasowego fermiona = zakaz (poprawny fizycznie).
K(r=0)=⊥ = osobliwość (rzeczywista). T=0=⊥ = granica III zasady.
Czy ⊥ ma różne znaczenia w różnych kontekstach?]*

### 4.2 Relationship to regularisation

Dimensional regularisation (DR) and ζ-regularisation assign finite values to expressions
that classically diverge — the same expressions that Wheel maps to ⊥.

```
Γ(ε) = ⊥   (Gamma function at non-positive integers)
ζ(1) = ⊥   (Riemann zeta at s = 1)
```

*[TODO — czy DR i ζ-regularyzacja są izomorficzne z Wheel w sensie formalnym?
Czy renormalizacja QED staje się zbędna przy Wheel? To jest daleko idące twierdzenie
— wymaga ostrożności.]*

### 4.3 Towards Wheel on Riemannian manifolds

*[TODO — otwarte pytanie: jak Wheel działa na rozmaitościach? Czy tensor metryczny
w algebrze koła ma naturalny odpowiednik? To kierunek do QG.]*

### 4.4 Limitations

- **Wheel operates pointwise.** UV divergences in QFT live in integrals ∫d⁴k, not
  at individual points. Wheel cannot directly address UV renormalisation.

- **Coordinate-dependent objects require invariants.** Christoffel symbols, metric
  components in fixed coordinates — their ⊥ may be an artefact. The framework
  requires pairing with tensorial invariants (Kretschmann, Ricci scalar) for
  physical interpretation.

- **Complex poles.** Wheel over ℝ misses poles with non-zero imaginary part.

- **Wheel ≠ limit theory** — this is a feature, not a bug, but it means the
  framework returns ⊥ for removable singularities. The analytic layer (`wheel_calculus`)
  is required to recover limits.

---

## 5. Related Work

*[TODO — Carlström (2004), Meyenburg (2023) + wyjaśnienie wyboru.
Meadows (1997) "Dividing by Nothing" — historyczny kontekst.
IEEE 754 NaN — porównanie (NaN nie jest absorpcyjny, nie jest algebraiczny).
Możliwe: Rosinger (2004/2008) — kwestia czy cytować.]*

- **Carlström, J. (2004).** Wheels — On Division by Zero. *[journal/proceedings TBD]*
  — foundational reference.

- **Meyenburg, T. (2023).** *[TODO — pełna referencja]* — semi-ring extension with
  directed infinities. We use Carlström; rationale in Section 2.4.

---

## 6. Conclusion

*[TODO — napisać po dopięciu. Powinno odpowiadać na Abstract.
Kluczowe zdanie: framework nie rozwiązuje osobliwości — klasyfikuje je algebraicznie
i zachowuje informację o ich typie. To jest krokiem w stronę teorii, która traktuje
osobliwości jako obiekty matematyczne, nie jako błędy obliczeniowe.]*

We have presented a symbolic algebra framework in which division by zero is a
well-defined algebraic operation. Applied to canonical equations of theoretical physics,
the framework identifies and classifies singularities consistently with known physical
results. Several non-trivial observations emerge: the algebraic isomorphism between
resonance and on-shell conditions, the role of curvature invariants in distinguishing
physical from coordinate singularities, and a formal connection between the ⊥ element
and standard regularisation procedures.

The framework is implemented in Python (SymPy), tested, and verifiable by running
`python main.py`.

---

## Appendix A: Singularity Type System

```
SingularityType enum (wheel_calculus.py)

  REGULAR       → ✓   regular point
  REMOVABLE     → ⊥→v  removable (Taylor recovers limit)
  POLE_SIMPLE   → ⊥   simple pole (order 1), residue defined
  POLE_HIGHER   → ⊥   higher-order pole, residue N/A
  POLE          → ⊥   generic pole (order unknown)
  ESSENTIAL     → ⊥   essential singularity (Picard)
  LOGARITHMIC   → ⊥   logarithmic divergence (QCD)
  BRANCH_POINT  → ⊥   branch point
  COORDINATE    → ⊥*  coordinate artefact (needs invariant)
  PHYSICAL      → ⊥   confirmed physical singularity
  COMPLEX_POLE  → ⊥?  complex pole (off real axis)
  UNKNOWN       → ?   fallback
```

Properties:
- `.is_genuine_singularity` — not removable
- `.has_residue` — Cauchy residue defined (POLE_SIMPLE only)
- `.short` — label for tables and logs

---

## Appendix B: Equation Database Summary

*[TODO — wygenerować z `python main.py --db` i wkleić / zautomatyzować.]*

---

## Appendix C: Code Availability

All results in this paper are reproducible by running:

```bash
git clone https://github.com/Mariusz-Rossa/WheelPhysics.git
cd WheelPhysics
pip install sympy numpy
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

DOI: [TODO: Zenodo DOI after upload]

---

*— End of draft v0.1 —*
*Sections marked [TODO] are placeholders for content to be written after
GitHub/Zenodo upload and further analysis.*