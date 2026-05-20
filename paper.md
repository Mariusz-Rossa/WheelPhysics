---
title: 'WheelPhysics: A Singularity-Aware Symbolic Algebra Framework for Physical Equations'
tags:
  - Python
  - wheel algebra
  - singularities
  - symbolic mathematics
  - general relativity
  - quantum field theory
  - SymPy
authors:
  - name: Mariusz Rossa
    orcid: 0009-0006-1060-2883
    affiliation: 1
affiliations:
  - name: Independent Researcher, Poland
    index: 1
date: 2026-05-20
bibliography: paper.bib
---

# Summary

Singular points in theoretical physics — the Schwarzschild singularity at $r = 0$,
the Big Bang at $t = 0$, infrared divergences in quantum field theory — are conventionally
treated as computational breakpoints where standard mathematics fails.
Computer algebra systems typically abort at these points, returning an error and
discarding any structural information about the nature of the singularity itself.

**WheelPhysics** is a Python framework that applies Wheel Algebra [@carlstrom2004] to
canonical equations of theoretical physics. Wheel Algebra extends a commutative ring with
a single absorbing element $\bot$ ("bottom") such that $x/0 = \bot$ for all $x$, including
$x = 0$. The element $\bot$ is not an error — it is a well-defined member of the algebra,
satisfying $\bot + x = \bot$ and $\bot \cdot x = \bot$ for all $x$. Expressions containing
$\bot$ remain algebraically manipulable.

The framework evaluates 46 canonical equations across six physical domains (general
relativity, cosmology, quantum field theory, classical mechanics, thermodynamics, and pure
mathematics), cataloguing 60 known singular points and classifying each algebraically.

# Statement of Need

Two problems motivate this work.

**The coordinate artefact problem.** In general relativity, the Schwarzschild metric
component $g_{rr} = (1 - r_s/r)^{-1}$ diverges at both $r = r_s$ (the event horizon)
and $r = 0$ (the geometric centre). Both points produce the same division-by-zero error
in classical algebra. Yet the two singularities are physically entirely different: the
horizon is a coordinate artefact, regular in Kruskal–Szekeres coordinates, while $r = 0$
is a genuine curvature singularity present in every coordinate system. Classical algebra
does not encode this distinction; it must be supplied by geometric argument.

**The Feynman prescription.** The scalar Feynman propagator $\Delta(p) = 1/(p^2 - m^2)$
is singular precisely at the on-shell condition $p^2 = m^2$, where a physical particle
propagates. The standard workaround — displacing the pole to $1/(p^2 - m^2 + i\varepsilon)$
— is procedurally justified but has no purely algebraic foundation: it amounts to
deliberately avoiding the point where the algebra breaks down.

Both problems share the same structure: a physically meaningful point is singular in the
algebra, and practitioners must handle it by external prescription rather than by the
algebra itself. WheelPhysics asks whether admitting $\bot$ as a first-class algebraic
object can make this structure explicit and classifiable.

# Framework Design

The framework is organised around a strict separation between two layers.

**The algebraic layer** (`wheel_algebra`) implements Wheel Algebra axiomatically. At any
singular point it returns $\bot$, unconditionally and without appeal to limits. This layer
knows nothing about whether a singularity is removable, what order a pole has, or whether
it is physical. It simply records that the algebra is undefined at this point.

**The analytic layer** (`wheel_calculus`) operates on top. When the algebraic layer returns
$\bot$, the analytic layer asks: *what kind of singularity is this?* It implements a
four-way classification:

$$
\text{Wheel result at } x_0: \quad
\begin{cases}
\text{Finite}(v) & \text{regular point} \\
\bot \xrightarrow{\text{0/0, Taylor}} \text{RemovableSingularity}(\lim = v) \\
\bot \xrightarrow{\text{true pole, Cauchy}} \text{PoleSingularity}(\text{order},\, \text{res},\, \text{Laurent}) \\
\bot & \text{irreducible (essential, logarithmic, ...)}
\end{cases}
$$

For true poles, the analytic layer computes the pole order $n$ by testing
$\lim_{x \to x_0}(x - x_0)^n f(x)$, and for simple poles ($n = 1$) computes the
Cauchy residue $\mathrm{res}(f, x_0) = \lim_{x \to x_0}(x - x_0) f(x)$.

A formal type system (`SingularityType`, 12 types) encodes physical distinctions:
coordinate artefacts (`COORDINATE`), confirmed physical singularities (`PHYSICAL`),
logarithmic divergences (`LOGARITHMIC`), and complex poles (`COMPLEX_POLE`), among others.

One non-trivial implementation issue deserves mention: SymPy's symbolic simplifier
can silently mask zeros in denominators before substitution, causing the framework to
miss a singularity. This is resolved by recursively inspecting all denominator
sub-expressions in the abstract syntax tree *before* any substitution takes place.

# Results

**Kretschmann invariant.** The Kretschmann scalar $K = 12 r_s^2 / r^6$ is a
coordinate-independent curvature invariant. Evaluating it at both Schwarzschild
singular points:

$$K(r = r_s) = \frac{12}{r_s^4} \quad \text{(finite)}, \qquad K(r = 0) = \bot$$

The framework correctly distinguishes the two singularities without external argument:
the horizon is algebraically regular under a tensorial invariant; the centre is not.
This is the cleanest demonstration that $\bot$ carries physically interpretable information.

**On-shell condition.** Every Feynman propagator in the database — scalar
$1/(p^2 - m^2)$, fermionic $(p + m)/(p^2 - m^2)$, photonic $1/k^2$, Compton amplitude
$1/(s - m^2)$ — returns $\bot$ exactly at the on-shell condition. Off-shell (virtual
particles): the propagator is finite and computable. The on-shell/off-shell distinction
maps cleanly onto the $\bot$/finite distinction in the algebra.

**Resonance–on-shell isomorphism.** Residue analysis reveals a structural identity:

$$\mathrm{res}\!\left(\frac{1}{p^2 - m^2},\, p = m\right) = \frac{1}{2m}, \qquad
\mathrm{res}\!\left(\frac{1}{\omega^2 - \omega_0^2},\, \omega = \omega_0\right) = \frac{1}{2\omega_0}$$

A classical harmonic oscillator at resonance and a relativistic quantum field theory
propagator at the on-shell condition are described by algebraically identical residue
formulae. This isomorphism — formally documented here via the Cauchy theorem — connects
classical and quantum mechanics at the level of pole structure.

**Removable singularities.** Wheel Algebra returns $\bot$ at every $0/0$ form, including
removable singularities. This is the correct algebraic behaviour: $\sin(x)/x$ at $x = 0$
is genuinely undefined as a ratio, even though the limit exists. The analytic layer
recovers the limit via Taylor expansion: $\sin(x)/x \to \bot \to \lim = 1$.
The sinc function, the Planck distribution as $T \to 0$, and the $x/(e^x - 1)$
factor in Bose–Einstein statistics are all handled correctly.

**Regularisation functions.** The Gamma function $\Gamma(\varepsilon)$ at
non-positive integers and the Riemann zeta function $\zeta(1)$ — both used as
regularising tools in dimensional regularisation and $\zeta$-regularisation — return
$\bot$ at their poles. This places standard QFT regularisation procedures in algebraic
contact with the $\bot$ element, suggesting a structural (rather than merely procedural)
relationship between Wheel Algebra and renormalisation.

**Summary statistics.** Of 46 equations and 60 singular points: $\bot$ at 38 points,
finite at 4, mixed (some singular points finite, others $\bot$) at 4 equations.
The consistency verification suite passes 100% across all expressions.

# Limitations

The framework operates **pointwise**: UV divergences in QFT live in loop integrals
$\int d^4k$, not at individual momentum values, and are outside the scope of this approach.
Coordinate-dependent objects (Christoffel symbols, metric components in fixed coordinates)
require pairing with tensorial invariants for physical interpretation.
Complex poles — such as those of the damped harmonic oscillator Green's function at
$\omega = \pm\sqrt{\omega_0^2 - \gamma^2/4} \pm i\gamma/2$ — lie off the real axis
and are not reached by real substitution; extension to $\mathbb{C}$ remains open.
Logarithmic divergences (QCD Landau pole) are correctly flagged as $\bot$ but require
asymptotic (Poincaré) series rather than Taylor expansion for further classification.

# Acknowledgements

No funding was received for this work.

# References