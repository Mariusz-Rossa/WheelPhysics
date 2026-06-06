# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
equations_db.py — database of physical equations

Catalogue of known equations with metadata:
  - priority (1=division, 2=singularity, 3=limit)
  - variables and critical values
  - known singularities and their physical meaning
  - Wheel analysis result (filled by the system)

Strategy according to project instructions:
  Prio 1: equations with division
  Prio 2: singularities (result → ∞)
  Prio 3: limits and derivatives (0/0)
  Skipped: without division (identical in Wheel)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import sympy as sp


class Priority(IntEnum):
    DIVISION    = 1   # contains division
    SINGULARITY = 2   # result → ∞ at certain points
    LIMIT       = 3   # 0/0 form or limit


@dataclass
class PhysicsEquation:
    """Description of a single equation in the database."""
    name:             str
    domain:           str                    # "GR", "QFT", "thermo", ...
    expression:       sp.Basic
    variables:        list[sp.Symbol]
    parameters:       list[sp.Symbol]
    priority:         Priority
    known_singular:   list[dict]             # [{var, value, description}]
    physical_meaning: str
    wheel_result:     Optional[str] = None   # filled after analysis
    notes:            str = ""

    def one_liner(self) -> str:
        prio_str = f"P{int(self.priority)}"
        sing_count = len(self.known_singular)
        return (
            f"[{prio_str}] [{self.domain:<5}] {self.name:<40} "
            f"| singularities: {sing_count}"
        )


# ─── Equations database ───────────────────────────────────────────────────────

def build_database() -> list[PhysicsEquation]:
    """Builds and returns the full database of equations."""

    # Symbols
    r, r_s         = sp.symbols("r r_s",             positive=True)
    theta          = sp.Symbol("theta",               positive=True)
    t_s            = sp.Symbol("t",                   positive=True)
    a              = sp.Symbol("a",                   positive=True)
    p, m, k, q     = sp.symbols("p m k q",            real=True)
    m_e            = sp.Symbol("m_e",                 positive=True)
    G, M, c        = sp.symbols("G M c",              positive=True)
    H              = sp.Symbol("H",                   real=True)
    Lambda         = sp.Symbol("Lambda",              positive=True)
    rho            = sp.Symbol("rho",                 positive=True)
    k_curv         = sp.Symbol("k_curv",              real=True)
    T              = sp.Symbol("T",                   positive=True)
    kB             = sp.Symbol("k_B",                 positive=True)
    omega, omega0  = sp.symbols("omega omega0",       positive=True)
    hbar           = sp.Symbol("hbar",                positive=True)
    epsilon        = sp.Symbol("epsilon",             positive=True)
    x              = sp.Symbol("x",                   real=True)
    n              = sp.Symbol("n",                   positive=True, integer=True)
    k_e            = sp.Symbol("k_e",                 positive=True)
    e_charge       = sp.Symbol("e_charge",            positive=True)
    q_mom          = sp.Symbol("q_mom",               real=True)
    V              = sp.Symbol("V",                   positive=True)
    s              = sp.Symbol("s",                   positive=True)

    db = []

    # ── PRIORITY 1: General Relativity ────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Schwarzschild g_rr",
        domain="GR",
        expression=1 / (1 - r_s / r),
        variables=[r],
        parameters=[r_s],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": r_s,        "description": "Event horizon (coordinate artifact)"},
            {"var": r, "value": sp.S.Zero,  "description": "Physical singularity"},
        ],
        physical_meaning="Radial component of the Schwarzschild metric tensor",
        notes="K(r_s) is finite — the horizon is NOT a physical singularity",
    ))

    db.append(PhysicsEquation(
        name="Schwarzschild g_tt",
        domain="GR",
        expression=-(1 - r_s / r) * c**2,
        variables=[r],
        parameters=[r_s, c],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": sp.S.Zero, "description": "Physical singularity r=0"},
        ],
        physical_meaning="Time component of the Schwarzschild metric tensor",
    ))

    db.append(PhysicsEquation(
        name="Kretschmann invariant",
        domain="GR",
        expression=12 * r_s**2 / r**6,
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero, "description": "True singularity — K→∞"},
        ],
        physical_meaning="K = R_abcd R^abcd — scalar curvature invariant. K(r_s) is finite.",
        notes="Key test: distinguishes coordinate artifact from physical singularity",
    ))

    db.append(PhysicsEquation(
        name="Christoffel symbol Γ^t_tr",
        domain="GR",
        expression=r_s / (2 * r * (r - r_s)),
        variables=[r],
        parameters=[r_s],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": r_s,       "description": "Horizon — coordinate singularity"},
            {"var": r, "value": sp.S.Zero, "description": "Physical singularity"},
        ],
        physical_meaning="Christoffel symbol — connection to geodesic acceleration",
    ))

    # ── PRIORITY 1: Cosmology ─────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Friedmann — curvature term",
        domain="COSMO",
        expression=k_curv * c**2 / a**2,
        variables=[a],
        parameters=[k_curv, c],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": a, "value": sp.S.Zero, "description": "Big Bang / Big Crunch"},
        ],
        physical_meaning="Curvature term in the Friedmann equation H² = 8πGρ/3 - kc²/a² + Λc²/3",
        notes="Symmetry a→-a suggests the existence of a 'pre-universe' in Wheel",
    ))

    db.append(PhysicsEquation(
        name="Matter density ρ~1/a³",
        domain="COSMO",
        expression=rho / a**3,
        variables=[a],
        parameters=[rho],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": a, "value": sp.S.Zero, "description": "Big Bang — density → ∞"},
        ],
        physical_meaning="Evolution of matter density with scale factor",
    ))

    db.append(PhysicsEquation(
        name="Radiation density ρ~1/a⁴",
        domain="COSMO",
        expression=rho / a**4,
        variables=[a],
        parameters=[rho],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": a, "value": sp.S.Zero, "description": "Big Bang — radiation → ∞"},
        ],
        physical_meaning="Evolution of radiation density (+ relativistic pressure term)",
    ))

    # ── PRIORITY 1: QFT ───────────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Feynman scalar propagator",
        domain="QFT",
        expression=sp.Integer(1) / (p**2 - m**2),
        variables=[p],
        parameters=[m],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value":  m, "description": "On-shell (p=+m) — real particle"},
            {"var": p, "value": -m, "description": "On-shell (p=-m) — antiparticle"},
        ],
        physical_meaning="Klein-Gordon propagator. Pole = asymptotic (observable) state.",
        notes="Hypothesis: ⊥ on-shell = algebraic definition of observability",
    ))

    db.append(PhysicsEquation(
        name="Photon propagator",
        domain="QFT",
        expression=sp.Integer(1) / k**2,
        variables=[k],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": k, "value": sp.S.Zero, "description": "Massless photon on-shell (k=0)"},
        ],
        physical_meaning="Photon propagator in Lorenz gauge. IR singularity at k=0.",
    ))

    db.append(PhysicsEquation(
        name="Fermion propagator (simplified)",
        domain="QFT",
        expression=(p + m_e) / (p**2 - m_e**2),
        variables=[p],
        parameters=[m_e],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value":  m_e, "description": "On-shell electron"},
            {"var": p, "value": -m_e, "description": "On-shell positron"},
        ],
        physical_meaning="Dirac propagator (scalarized). Numerator: p̸+m after trace.",
    ))

    db.append(PhysicsEquation(
        name="Massless fermion propagator",
        domain="QFT",
        expression=sp.Integer(1) / p,
        variables=[p],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value": sp.S.Zero, "description": "IR — massless fermion cannot have p=0"},
        ],
        physical_meaning="Dirac propagator for m=0 (neutrinos, chiral quarks).",
        notes="Wheel algebraically forbids p=0 for massless particles — physically correct",
    ))

    # ── PRIORITY 2: Thermodynamics / statistical ──────────────────────────────

    db.append(PhysicsEquation(
        name="Boltzmann distribution 1/T",
        domain="THERMO",
        expression=sp.exp(-epsilon / (kB * T)) / T,
        variables=[T],
        parameters=[epsilon, kB],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": T, "value": sp.S.Zero, "description": "Absolute zero — thermodynamic singularity"},
        ],
        physical_meaning="Probability of occupying energy state ε at temp. T",
    ))

    db.append(PhysicsEquation(
        name="Planck distribution",
        domain="THERMO",
        expression=hbar * omega / (sp.exp(hbar * omega / (kB * T)) - 1),
        variables=[T],
        parameters=[hbar, omega, kB],
        priority=Priority.LIMIT,
        known_singular=[
            {"var": T, "value": sp.S.Zero, "description": "T=0 — denominator exp(∞)-1 → ∞"},
        ],
        physical_meaning="Energy of photons in black body radiation",
        notes="0/0 form when T→∞ (classical Rayleigh-Jeans limit)",
    ))

    # ── PRIORITY 3: Limits and derivatives ────────────────────────────────────

    db.append(PhysicsEquation(
        name="sinc(x) = sin(x)/x",
        domain="MATH",
        expression=sp.sin(x) / x,
        variables=[x],
        parameters=[],
        priority=Priority.LIMIT,
        known_singular=[
            {"var": x, "value": sp.S.Zero, "description": "0/0 form — limit = 1"},
        ],
        physical_meaning="sinc function — appears in diffraction, Fourier transform",
        notes="Classically: lim(x→0) sin(x)/x = 1. Wheel: ⊥. This is a significant difference!",
    ))

    # ── NEW: Riemann Tensor — Schwarzschild ───────────────────────────────────

    db.append(PhysicsEquation(
        name="Riemann tensor R^r_trt",
        domain="GR",
        expression=-r_s / r**3,
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Physical singularity — curvature → ∞"},
        ],
        physical_meaning=(
            "R^r_trt = -r_s/r³ — Riemann tensor component for Schwarzschild. "
            "At r=r_s: -1/r_s² (finite — regular horizon). "
            "At r=0: ⊥ (physical singularity)."
        ),
        notes="R(r_s) finite — Riemann tensor confirms the horizon is an artifact",
    ))

    db.append(PhysicsEquation(
        name="Riemann tensor R^θ_rθr",
        domain="GR",
        expression=-r_s / (2 * r**3),
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Physical singularity — angular curvature → ∞"},
        ],
        physical_meaning=(
            "R^θ_rθr = -r_s/(2r³) — angular-radial component. "
            "Measures curvature in angular directions. "
            "At r=r_s: -1/(2r_s²) is finite. At r=0: ⊥."
        ),
        notes="Same structure as R^r_trt — singularity only at r=0",
    ))

    db.append(PhysicsEquation(
        name="Riemann tensor R^φ_tφt",
        domain="GR",
        expression=(r_s / r**3) * (1 - r_s / r),
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Physical singularity"},
            {"var": r, "value": r_s,
             "description": "Horizon — factor f(r)=0 cancels r_s/r³, result 0 not ⊥"},
        ],
        physical_meaning=(
            "R^φ_tφt = (r_s/r³)·(1-r_s/r). "
            "Exceptional: at r=r_s we have 0·∞ — f(r)→0 but r_s/r³→∞. "
            "Wheel through recursion on /r in f(r) gives ⊥ at r=0. "
            "At r=r_s: (1-r_s/r)→0 cancels divergence, result = 0."
        ),
        notes="0·∞ case at r=r_s — physically V_eff=0 at the horizon",
    ))

    # ── NEW: Dirac Equation ───────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Dirac m=0 — Weyl propagator",
        domain="QFT",
        expression=sp.Integer(1) / p,
        variables=[p],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value": sp.S.Zero,
             "description": "Massless fermion on-shell — p=0 physically unattainable"},
        ],
        physical_meaning=(
            "lim(m→0) S_F(p,m) = 1/p. "
            "Dirac equation for m=0 is two independent Weyl equations. "
            "Pole at p=0 — state of rest for massless particle is inaccessible (moves at c)."
        ),
        notes="Wheel algebraically derives the prohibition of p=0 for massless particles",
    ))

    db.append(PhysicsEquation(
        name="Dirac — relativistic energy 1/√(p²+m²)",
        domain="QFT",
        expression=sp.Integer(1) / sp.sqrt(p**2 + m**2),
        variables=[p, m],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": m, "value": sp.S.Zero,
             "description": "Massless limit at p=0 — 1/|p|, IR singularity"},
        ],
        physical_meaning=(
            "1/E = 1/√(p²+m²) — relativistic state normalization. "
            "At m=0 and p=0: 1/0 → ⊥. "
            "At m>0 and p=0: 1/m (finite — mass regularizes IR)."
        ),
        notes="Wheel distinguishes: m>0 no singularity at p=0, m=0 gives ⊥",
    ))

    # ── NEW: Klein-Gordon in curved spacetime ─────────────────────────────────

    l_sym = sp.Symbol("l", nonneg=True, integer=True)

    db.append(PhysicsEquation(
        name="Klein-Gordon in Schwarzschild V_eff",
        domain="QFT",
        expression=(1 - r_s/r) * (m**2 + l_sym*(l_sym+1)/r**2 + r_s/r**3),
        variables=[r],
        parameters=[r_s, m, l_sym],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Physical singularity — V_eff → ∞"},
            {"var": r, "value": r_s,
             "description": "Horizon — f(r)=0 zeroes V_eff (barrier disappears)"},
        ],
        physical_meaning=(
            "V_eff = f(r)·[m² + l(l+1)/r² + r_s/r³], f=1-r_s/r. "
            "Effective potential of KG in Schwarzschild (tortoise coords). "
            "First connection of GR+QFT — both singularities overlap at r=0. "
            "At r=r_s: V_eff=0 (horizon = barrier disappears — physically correct). "
            "At r=0: V_eff=⊥."
        ),
        notes="r=r_s gives V_eff=0, not ⊥ — the horizon here is a vanishing barrier, not a singularity",
    ))

    db.append(PhysicsEquation(
        name="Euclidean Klein-Gordon 1/(p²+m²)",
        domain="QFT",
        expression=sp.Integer(1) / (p**2 + m**2),
        variables=[p],
        parameters=[m],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value": sp.S.Zero,
             "description": "At m=0: IR pole at p=0"},
        ],
        physical_meaning=(
            "Euclidean KG propagator: 1/(p²+m²). "
            "For m>0: no pole on the real axis — hence the utility of Wick rotation. "
            "For m=0 at p=0: ⊥. "
            "Comparison with Minkowski 1/(p²-m²): Wick rotation removes real poles."
        ),
        notes="m=0,p=0 → ⊥. For m>0 and p=0: 1/m² finite",
    ))


    # ── GR: Kerr Metric ───────────────────────────────────────────────────────

    a_kerr = sp.Symbol("a_kerr", positive=True)   # angular momentum / mass
    Delta  = r**2 - r_s*r + a_kerr**2             # Kerr function
    Sigma  = r**2 + a_kerr**2 * sp.cos(theta)**2  # shape factor

    db.append(PhysicsEquation(
        name="Kerr g_rr — Δ(r) in denominator",
        domain="GR",
        expression=Sigma / Delta,
        variables=[r],
        parameters=[r_s, a_kerr, theta],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": r_s/2 + sp.sqrt(r_s**2/4 - a_kerr**2),
             "description": "Outer horizon r+ — Δ(r+)=0"},
            {"var": r, "value": r_s/2 - sp.sqrt(r_s**2/4 - a_kerr**2),
             "description": "Inner horizon r- — Δ(r-)=0"},
        ],
        physical_meaning=(
            "g_rr component of Kerr metric (rotating black hole). "
            "Δ = r²-r_s·r+a² — zeroes out at two horizons r±. "
            "Kerr ring singularity: r=0, θ=π/2 (Σ→0 and Δ→a²≠0). "
            "More realistic model than Schwarzschild — every astrophysical BH rotates."
        ),
        notes="Two horizons instead of one — richer singularity structure than Schwarzschild",
    ))

    # ── GR: Reissner-Nordström Metric ─────────────────────────────────────────

    r_Q = sp.Symbol("r_Q", positive=True)   # charge radius: r_Q²=GQ²/(4πε₀c⁴)
    f_RN = 1 - r_s/r + r_Q**2/r**2

    db.append(PhysicsEquation(
        name="Reissner-Nordström g_rr",
        domain="GR",
        expression=sp.Integer(1) / f_RN,
        variables=[r],
        parameters=[r_s, r_Q],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": r_s/2 + sp.sqrt(r_s**2/4 - r_Q**2),
             "description": "Outer horizon r+ — charged BH"},
            {"var": r, "value": r_s/2 - sp.sqrt(r_s**2/4 - r_Q**2),
             "description": "Inner horizon r- (Cauchy)"},
            {"var": r, "value": sp.S.Zero,
             "description": "Physical singularity r=0"},
        ],
        physical_meaning=(
            "g_rr of Reissner-Nordström metric (black hole with charge Q). "
            "f_RN = 1 - r_s/r + r_Q²/r². Three singularities: r+, r-, r=0. "
            "When r_Q = r_s/2: extremal horizon (r+=r-). "
            "When r_Q > r_s/2: naked singularities (no horizon)."
        ),
        notes="Three singularities — richest structure among spherical metrics",
    ))

    # ── GR: Hubble Radius ─────────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Hubble radius r_H = c/H",
        domain="COSMO",
        expression=c / H,
        variables=[H],
        parameters=[c],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": H, "value": sp.S.Zero,
             "description": "H=0 — static universe, no cosmological horizon"},
        ],
        physical_meaning=(
            "r_H = c/H — size of the Hubble horizon. "
            "H=0: static universe (Einstein model), horizon → ∞. "
            "In Wheel: c/0 = ⊥. "
            "Connection to Friedmann: H² → 0 when a → constant."
        ),
        notes="Connects to Friedmann equations — when H²=0, r_H=⊥",
    ))

    # ── Hawking Temperature ───────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Hawking temperature T_H",
        domain="GR",
        expression=hbar * c**3 / (8 * sp.pi * G * M * kB),
        variables=[M],
        parameters=[hbar, c, G, kB],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": M, "value": sp.S.Zero,
             "description": "M=0 — no black hole, T_H → ∞"},
        ],
        physical_meaning=(
            "T_H = ħc³/(8πGMk_B) — temperature of Hawking radiation. "
            "The smaller the BH mass, the higher the temperature (paradox). "
            "M→0: T_H→∞ — final stage of BH evaporation. "
            "Wheel: T_H(M=0) = ⊥. Connection to Kretschmann: K~1/r⁶, M~r_s."
        ),
        notes="Connects GR with QFT — thermal radiation from event horizon",
    ))

    # ── Classical Mechanics ───────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Coulomb potential ke²/r",
        domain="CLASS",
        expression=k_e * e_charge**2 / r,
        variables=[r],
        parameters=[k_e, e_charge],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Point charge — potential → ∞"},
        ],
        physical_meaning=(
            "V_C = ke²/r — electrostatic potential of a point charge. "
            "Archetype of all 1/r singularities in physics. "
            "In QED replaced by photon propagator 1/q² (already in db). "
            "Wheel: V_C(r=0) = ⊥ — point charge is a singularity."
        ),
        notes="Archetype of 1/r singularities — foundation of classical electrodynamics",
    ))

    db.append(PhysicsEquation(
        name="Gravitational potential -GM/r",
        domain="CLASS",
        expression=-G * M / r,
        variables=[r],
        parameters=[G, M],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Point mass — potential → -∞"},
        ],
        physical_meaning=(
            "V_g = -GM/r — Newtonian gravitational potential. "
            "Non-relativistic limit of Schwarzschild metric (g_tt ≈ -1 + r_s/r). "
            "Wheel: V_g(r=0) = ⊥. "
            "Same singularity as in GR — Wheel is consistent in both limits."
        ),
        notes="Consistency: same ⊥ as in Schwarzschild for r=0",
    ))

    db.append(PhysicsEquation(
        name="Kepler force -GM/r²",
        domain="CLASS",
        expression=-G * M / r**2,
        variables=[r],
        parameters=[G, M],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Collision — force → ∞ (collisional singularity)"},
        ],
        physical_meaning=(
            "F = -GM/r² — Newton's gravity force / Kepler force. "
            "Collisional singularity at r=0 — foundation of the N-body problem. "
            "Wheel: F(r=0) = ⊥. "
            "Connection to ChaosEngine — in CRT collisional singularities are the same ⊥."
        ),
        notes="Collisional singularities in CRT — bridge between WheelPhysics and ChaosEngine",
    ))

    db.append(PhysicsEquation(
        name="Oscillator resonance 1/(ω²-ω₀²)",
        domain="CLASS",
        expression=sp.Integer(1) / (omega**2 - omega0**2),
        variables=[omega],
        parameters=[omega0],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": omega, "value": omega0,
             "description": "Resonance — amplitude → ∞ (without damping)"},
        ],
        physical_meaning=(
            "Amplitude of a forced harmonic oscillator: A ~ 1/(ω²-ω₀²). "
            "Resonance at ω=ω₀: A→∞ (without damping). "
            "Same structure as propagator: 1/(p²-m²) ↔ 1/(ω²-ω₀²). "
            "Wheel: A(ω=ω₀) = ⊥. Resonance = 'on-shell' state of the oscillator."
        ),
        notes="Isomorphism with QFT propagator: classical resonance = quantum on-shell",
    ))

    # ── Thermodynamics ────────────────────────────────────────────────────────

    a_vdw, b_vdw, R_gas = sp.symbols("a_vdw b_vdw R_gas", positive=True)

    db.append(PhysicsEquation(
        name="van der Waals — pressure P(V,T)",
        domain="THERMO",
        expression=R_gas * T / (V - b_vdw) - a_vdw / V**2,
        variables=[V],
        parameters=[T, R_gas, a_vdw, b_vdw],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": V, "value": b_vdw,
             "description": "V=b — proper volume of molecules (hard cores)"},
            {"var": V, "value": sp.S.Zero,
             "description": "V=0 — nonphysical singularity (gas cannot collapse)"},
        ],
        physical_meaning=(
            "P = RT/(V-b) - a/V² — van der Waals equation of state. "
            "V=b: molecules touch, pressure → ∞ (hard cores). "
            "V=0: purely mathematical singularity, nonphysical. "
            "Wheel correctly gives ⊥ at both points."
        ),
        notes="V=b is physical limit of real gas, V=0 is a mathematical artifact",
    ))

    Tc_sym = sp.Symbol("Tc", positive=True)

    db.append(PhysicsEquation(
        name="Specific heat at phase transition ~1/|T-Tc|",
        domain="THERMO",
        expression=sp.Integer(1) / sp.Abs(T - Tc_sym),
        variables=[T],
        parameters=[Tc_sym],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": T, "value": Tc_sym,
             "description": "T=Tc — critical point, C → ∞ (critical exponent α)"},
        ],
        physical_meaning=(
            "C ~ |T-Tc|^(-α) — specific heat divergence at phase transition. "
            "α ≈ 0.11 for 3D Ising, α=0 (log) for He-4. "
            "Simplification: α=1 (mean field). "
            "Wheel: C(T=Tc) = ⊥ — critical point is a thermodynamic singularity."
        ),
        notes="Critical exponents describe how fast we approach ⊥",
    ))

    # ── QFT: additional amplitudes ────────────────────────────────────────────

    s_var = sp.Symbol("s", positive=True)   # Mandelstam variable

    db.append(PhysicsEquation(
        name="Compton amplitude 1/(s-m²)",
        domain="QFT",
        expression=sp.Integer(1) / (s_var - m**2),
        variables=[s_var],
        parameters=[m],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": s_var, "value": m**2,
             "description": "s=m² — s-channel pole (intermediate particle on-shell)"},
        ],
        physical_meaning=(
            "Compton scattering amplitude: A ~ 1/(s-m²). "
            "s = (p+k)² — Mandelstam variable. "
            "Pole at s=m²: intermediate particle becomes real (on-shell). "
            "Identical structure to scalar propagator — ⊥ = asymptotic particle."
        ),
        notes="Confirms: poles in Mandelstam variables mean on-shell, i.e. ⊥",
    ))

    db.append(PhysicsEquation(
        name="QED photon exchange 1/q²",
        domain="QFT",
        expression=sp.Integer(1) / q_mom**2,
        variables=[q_mom],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": q_mom, "value": sp.S.Zero,
             "description": "q=0 — zero-momentum photon (IR, Coulomb potential)"},
        ],
        physical_meaning=(
            "Photon exchange amplitude in QED: M ~ e²/q². "
            "Limit q→0: Coulomb potential (we recover classical physics). "
            "Wheel: M(q=0) = ⊥. "
            "Connection between QED and classical electrodynamics via limit q→0."
        ),
        notes="q→0 is the classical limit of QED — Wheel gives ⊥ where classically V_C=∞",
    ))

    # ── Mathematics ───────────────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Gamma function Γ(n) ~ 1/n as n→0",
        domain="MATH",
        expression=sp.Integer(1) / n,
        variables=[n],
        parameters=[],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": n, "value": sp.S.Zero,
             "description": "n=0 — first pole of Γ(n), residue = 1"},
        ],
        physical_meaning=(
            "Γ(n) ~ 1/n as n→0 (and at n=-1,-2,...). "
            "In QFT dimensional regularization: Γ(ε) ~ 1/ε as ε→0. "
            "This is the source of UV divergences in dim reg! "
            "Wheel: Γ_pole(n=0) = ⊥. "
            "Hypothesis: dimensional regularization is a substitute for Wheel in integrals."
        ),
        notes="Bridge between Wheel and dimensional regularization — Γ(ε)=⊥ as ε→0",
    ))

    db.append(PhysicsEquation(
        name="Riemann ζ function — pole at s=1",
        domain="MATH",
        expression=sp.Integer(1) / (s - sp.Integer(1)),
        variables=[s],
        parameters=[],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": s, "value": sp.Integer(1),
             "description": "s=1 — pole of ζ(s), residue = 1"},
        ],
        physical_meaning=(
            "ζ(s) ~ 1/(s-1) as s→1 — the only pole of the Riemann function. "
            "Wheel: ζ_pole(s=1) = ⊥. "
            "In physics: ζ-regularization of sum 1+2+3+... = -1/12 bypasses the pole. "
            "Hypothesis: ζ-regularization is a classical substitute for Wheel for divergent series."
        ),
        notes="ζ-regularization and Wheel — two different ways for the same singularity",
    ))

    db.append(PhysicsEquation(
        name="Fourier transform IR — 1/ω",
        domain="MATH",
        expression=sp.Integer(1) / omega,
        variables=[omega],
        parameters=[],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": omega, "value": sp.S.Zero,
             "description": "ω=0 — IR divergence in frequency space"},
        ],
        physical_meaning=(
            "F[1/t](ω) ~ 1/ω — Fourier transform of 1/t function. "
            "IR divergence at ω=0 appears in: "
            "acoustics (zero modes), QFT (soft photons/gluons), "
            "turbulence (Kolmogorov spectrum). "
            "Wheel: (1/ω)(ω=0) = ⊥."
        ),
        notes="IR divergences in QFT and Wheel — ω=0 means no energy, unattainable state",
    ))


    # ══════════════════════════════════════════════════════════════════════════
    # NEW EQUATIONS — v0.9
    # GR/COSMO: Milne/de Sitter, ADM, photon V_eff
    # QFT: QCD gluon propagator, damped oscillator Green, Mandelstam t-channel
    # THERMO: Bekenstein-Hawking entropy (derivative), van Hove D(E)
    # MATH: sinc², (1-cos)/x² — counterexamples for wheel_calculus.py
    # ══════════════════════════════════════════════════════════════════════════

    # ── GR/COSMO: de Sitter Metric — cosmological horizon ─────────────────────

    db.append(PhysicsEquation(
        name="de Sitter metric — g_rr = 1/(1 - r²/R_H²)",
        domain="COSMO",
        expression=sp.Integer(1) / (1 - r**2 / r_s**2),   # r_s plays the role of R_H
        variables=[r],
        parameters=[r_s],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": r_s,
             "description": "r = R_H — de Sitter cosmological horizon (equivalent of r_s in Schwarzschild)"},
        ],
        physical_meaning=(
            "g_rr of de Sitter metric: 1/(1 - r²/R_H²) where R_H = c/H = √(3/Λ). "
            "Cosmological horizon at r=R_H — exact analogy to Schwarzschild horizon. "
            "Key difference: here the interior (r<R_H) is accessible to observer, "
            "and exterior (r>R_H) is behind the horizon (reverse of BH). "
            "H can change sign in more general models → two horizons. "
            "Wheel: g_rr(r=R_H) = ⊥ — same algebra as Schwarzschild."
        ),
        notes="Isomorphism Schwarzschild ↔ de Sitter: same ⊥ algebraic structure, reversed physics",
    ))

    # ── GR: ADM Energy — singularity at boundary (r→∞) ────────────────────────

    db.append(PhysicsEquation(
        name="ADM energy — 1/r term as r→∞",
        domain="GR",
        expression=sp.Integer(1) / r,
        variables=[r],
        parameters=[],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "r=0 — central singularity (here as integral singularity at r→∞ is reverse problem)"},
        ],
        physical_meaning=(
            "ADM Energy (Arnowitt-Deser-Misner) — integral energy of the system in GR. "
            "E_ADM = -(c²/16πG) ∮ (∂_j h_ii - ∂_i h_ij) dS^j. "
            "Asymptotically: h_ij ≈ δ_ij(1 + 2GM/rc²) — 1/r term dominates as r→∞. "
            "REVERSED PROBLEM: singularity not in center (r=0) but at integration boundary. "
            "Wheel operates pointwise — 1/r at r=0 gives ⊥, at r→∞ the 1/r term→0 (regular). "
            "Contrast with previous: here ⊥ in center is not ADM problem, "
            "but behavior at r→∞ determines the energy."
        ),
        notes="Reversed problem: physics in limit r→∞, not r→0. Wheel works pointwise — different logic.",
    ))

    # ── GR: Effective potential for photons (Schwarzschild photosphere) ───────

    db.append(PhysicsEquation(
        name="Photon potential V_ph = (1-r_s/r)/r²",
        domain="GR",
        expression=(1 - r_s / r) / r**2,
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "r=0 — physical singularity (V_ph → ∞)"},
            {"var": r, "value": r_s,
             "description": "r=r_s — horizon, f(r)=0 cancels 1/r², result V_ph=0"},
        ],
        physical_meaning=(
            "V_ph = f(r)/r² = (1-r_s/r)/r² — effective potential for photons (l=0, m=0) "
            "in Schwarzschild (tortoise coordinate). "
            "Photosphere (unstable circular photon orbit) at r_ph = 3r_s/2, "
            "where dV_ph/dr = 0: V_ph(r_ph) = 4/(27r_s²) — finite. "
            "Complements Klein-Gordon V_eff for massless particles (m=0, l=0). "
            "At r=r_s: V_ph = 0 (barrier disappears at horizon — analogy to KG). "
            "At r=0: V_ph = ⊥ (physical singularity)."
        ),
        notes="Complements KG V_eff for massless — m=0, l=0. Photosphere r_ph=3r_s/2 is regular.",
    ))

    # ── QFT: Gluon propagator with QCD self-interaction ───────────────────────

    alpha_s = sp.Symbol("alpha_s", positive=True)   # QCD coupling constant
    mu_r    = sp.Symbol("mu_r",    positive=True)   # renormalization scale
    k2      = sp.Symbol("k2",      positive=True)   # k² (momentum²)

    db.append(PhysicsEquation(
        name="QCD gluon propagator with loop correction",
        domain="QFT",
        expression=sp.Integer(1) / (k2 * (1 + alpha_s * sp.log(k2 / mu_r**2))),
        variables=[k2],
        parameters=[alpha_s, mu_r],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": k2, "value": sp.S.Zero,
             "description": "k²=0 — IR pole (zero-momentum photon, as in QED)"},
            {"var": k2, "value": mu_r**2 * sp.exp(-sp.Integer(1) / alpha_s),
             "description": "QCD Landau pole — k²=μ²·exp(-1/αs), non-perturbative"},
        ],
        physical_meaning=(
            "Gluon propagator with one-loop correction in QCD: 1/(k²(1+αs·log(k²/μ²))). "
            "TWO types of poles — both unknown to Wheel so far: "
            "(1) k²=0: standard IR pole like in QED (expected ⊥). "
            "(2) Landau pole: k²=μ²·exp(-1/αs) — LOGARITHMIC pole, "
            "different type than algebraic poles 1/xⁿ. "
            "In QED counterpart is unphysical (10^280 GeV). "
            "In QCD Landau pole is non-perturbative — appears at confinement scale (~ΛQCD). "
            "Question: does Wheel correctly detect logarithmic poles? Unexplored territory."
        ),
        notes="NEW TYPE: logarithmic pole. Wheel tested only on algebraic poles so far.",
    ))

    # ── QFT: Damped oscillator Green function (bridge resonance↔QM) ───────────

    gamma_d = sp.Symbol("gamma_d", positive=True)   # damping coefficient

    db.append(PhysicsEquation(
        name="Damped oscillator Green G(ω) = 1/(ω²-ω₀²+iγω)",
        domain="QFT",
        expression=sp.Integer(1) / (omega**2 - omega0**2 + sp.I * gamma_d * omega),
        variables=[omega],
        parameters=[omega0, gamma_d],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": gamma_d, "value": sp.S.Zero,
             "description": "γ→0 — pole returns to real axis: ω=±ω₀ (classical resonance)"},
            {"var": omega, "value": sp.I * gamma_d / 2 + sp.sqrt(omega0**2 - gamma_d**2 / 4),
             "description": "Pole in upper complex half-plane (for γ>0)"},
        ],
        physical_meaning=(
            "G(ω) = 1/(ω²-ω₀²+iγω) — Green function of damped oscillator. "
            "Pole moves to complex plane when γ>0: "
            "ω_± = ±√(ω₀²-γ²/4) + iγ/2. "
            "BRIDGE between three objects: "
            "(1) γ=0: classical resonance 1/(ω²-ω₀²) — already in db. "
            "(2) γ→0⁺: Feynman's iε prescription! (iγω acts as iε). "
            "(3) γ>0: physical resonance width = quantum state lifetime (Breit-Wigner rule). "
            "Wheel operates on real numbers — ω is real. "
            "Pole is complex → Wheel will NOT hit it via real substitution. "
            "This opens a question: how does Wheel deal with complex poles?"
        ),
        notes="CRITICAL: γ→0 is continuous transition to resonance↔iε. Complex poles are new territory for Wheel.",
    ))

    # ── QFT: Mandelstam t-channel — singularity at momentum transfer t=0 ──────

    t_man = sp.Symbol("t_man", real=True)   # Mandelstam variable t

    db.append(PhysicsEquation(
        name="t-channel amplitude 1/(t-m²)",
        domain="QFT",
        expression=sp.Integer(1) / (t_man - m**2),
        variables=[t_man],
        parameters=[m],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": t_man, "value": m**2,
             "description": "t=m² — exchanged particle on-shell (Coulomb limit as m→0, t→0)"},
            {"var": t_man, "value": sp.S.Zero,
             "description": "t=0 at m=0 — Coulomb limit: exchange of massless boson at zero momentum"},
        ],
        physical_meaning=(
            "Amplitude in t-channel 2→2 scattering: M ~ 1/(t-m²), t=(p1-p3)². "
            "t is momentum transfer — always t≤0 for physical scattering. "
            "DIFFERENCE from s-channel (Compton amplitude): "
            "s > 0 (center-of-mass energy), t ≤ 0 (momentum transfer). "
            "As m→0: pole at t=0 — QED Coulomb limit (1/q² potential). "
            "Wheel: 1/(t-m²) at t=m² → ⊥. At t=0, m=0 → ⊥. "
            "Complements s-channel (Compton amplitude) — full picture of Mandelstam variables s,t."
        ),
        notes="Complements s-channel: now we have s and t. u-channel = s+t+u=Σm² — adds no new structure.",
    ))

    # ── THERMO: Bekenstein-Hawking Entropy — derivative dS/dM ─────────────────

    db.append(PhysicsEquation(
        name="BH entropy — dS/dM = 1/T_H ~ M",
        domain="THERMO",
        expression=8 * sp.pi * G * M * kB / (hbar * c**3),
        variables=[M],
        parameters=[G, kB, hbar, c],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": M, "value": sp.S.Zero,
             "description": "M=0 — no black hole, dS/dM → 0 (but T_H=1/(dS/dM)→⊥)"},
        ],
        physical_meaning=(
            "Bekenstein-Hawking Entropy: S_BH = A/(4l_P²) = 4πGM²k_B/(ħc). "
            "dS/dM = 8πGMk_B/(ħc³) = 1/T_H — thermodynamic definition of temperature. "
            "S_BH itself is finite and well-defined (no singularity in S). "
            "But: T_H = (dS/dM)^(-1) = ħc³/(8πGMk_B) → ⊥ at M=0. "
            "CONNECTION to Hawking temperature (already in db): dS/dM = 1/T_H. "
            "dS/dM → 0 as M→0: entropy grows slower and slower during evaporation. "
            "Wheel: dS/dM(M=0) = 0 (finite!), but T_H = 1/(dS/dM) = ⊥. "
            "This shows how ⊥ propagates through inversion: finite → ⊥ via /0."
        ),
        notes="dS/dM is FINITE at M=0 (=0). But 1/(dS/dM)=T_H=⊥. Exercise with ⊥ propagation.",
    ))

    # ── THERMO: van Hove Density of States ────────────────────────────────────

    E_sym  = sp.Symbol("E",   real=True)
    E_c    = sp.Symbol("E_c", real=True)   # critical energy

    db.append(PhysicsEquation(
        name="van Hove density of states D(E) ~ 1/√|E-E_c|",
        domain="THERMO",
        expression=sp.Integer(1) / sp.sqrt(sp.Abs(E_sym - E_c)),
        variables=[E_sym],
        parameters=[E_c],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": E_sym, "value": E_c,
             "description": "E=E_c — van Hove point: energy band curvature = 0, D(E)→∞"},
        ],
        physical_meaning=(
            "D(E) ~ 1/√|E-E_c| — density of states at van Hove points in a solid. "
            "Appears when ∇_k E(k) = 0 (flat bottom or top of a band). "
            "Physically: infinite density of states = accumulation of orbitals with same energy. "
            "Direct consequence for: superconductivity (BCS), van Hove effect in optics, "
            "specific heat anomalies. "
            "COMPARISON with specific heat ~1/|T-Tc|: "
            "the latter diverges as 1/x (exponent α=1, mean field), "
            "this one as 1/√x (exponent α=1/2 — different universality class). "
            "Wheel: D(E_c) = ⊥ — instability point in band structure."
        ),
        notes="Exponent 1/2 (van Hove) vs 1 (mean field specific heat) — Wheel treats identically: ⊥.",
    ))

    # ── MATH: sinc²(x) = sin²(x)/x² — quadratic counterexample ────────────────

    db.append(PhysicsEquation(
        name="sinc²(x) = sin²(x)/x²",
        domain="MATH",
        expression=sp.sin(x)**2 / x**2,
        variables=[x],
        parameters=[],
        priority=Priority.LIMIT,
        known_singular=[
            {"var": x, "value": sp.S.Zero,
             "description": "0/0 form — limit = 1 (same as sinc, because lim sinc²=lim²sinc=1²=1)"},
        ],
        physical_meaning=(
            "sinc²(x) = sin²(x)/x² — square of sinc function. "
            "Appears in: diffraction intensity through a slit (I ~ sinc²), "
            "power spectrum of a rectangular signal, signal correlation. "
            "Limit: lim(x→0) sin²(x)/x² = [lim sinc(x)]² = 1² = 1. "
            "Wheel: sin²(0)/0² = 0/0 → ⊥. "
            "TEST for wheel_calculus.py: does Taylor module generalize to powers? "
            "sin(x) ≈ x - x³/6 → sin²(x) ≈ x² - x⁴/3 → sin²(x)/x² ≈ 1 - x²/3 → 1. "
            "Wheel_calculus should return 1, not ⊥."
        ),
        notes="Counterexample #2 for wheel_calculus.py — higher order Taylor. Result should be: ⊥→1.",
    ))

    # ── MATH: (1-cos(x))/x² — cosine counterexample ───────────────────────────

    db.append(PhysicsEquation(
        name="(1 - cos(x))/x²",
        domain="MATH",
        expression=(1 - sp.cos(x)) / x**2,
        variables=[x],
        parameters=[],
        priority=Priority.LIMIT,
        known_singular=[
            {"var": x, "value": sp.S.Zero,
             "description": "0/0 form — limit = 1/2 (different result than sinc!)"},
        ],
        physical_meaning=(
            "(1-cos(x))/x² — appears in: "
            "kinetic energy expansion (1-cos(ka)) in tight-binding model, "
            "phase accumulated by a qubit (geometric Berry phase), "
            "corrections to wave dispersion. "
            "Limit: lim(x→0) (1-cos(x))/x² = 1/2. "
            "Proof: cos(x) ≈ 1 - x²/2 + x⁴/24 → 1-cos(x) ≈ x²/2 → (1-cos)/x² → 1/2. "
            "Wheel: (1-cos(0))/0² = 0/0 → ⊥. "
            "KEY DIFFERENCE from sinc and sinc²: limit ≠ 1, limit = 1/2. "
            "Test for wheel_calculus.py: will module return 1/2 and not ⊥?"
        ),
        notes="Counterexample #3 for wheel_calculus.py — limit = 1/2, not 1. Different Taylor expansion.",
    ))

    return db


# ─── Database interface ───────────────────────────────────────────────────────

class EquationsDB:
    """Interface for physical equations database."""

    def __init__(self):
        self._db = build_database()

    def all(self) -> list[PhysicsEquation]:
        return self._db

    def by_priority(self, p: Priority) -> list[PhysicsEquation]:
        return [eq for eq in self._db if eq.priority == p]

    def by_domain(self, domain: str) -> list[PhysicsEquation]:
        return [eq for eq in self._db if eq.domain == domain.upper()]

    def with_singularities(self) -> list[PhysicsEquation]:
        return [eq for eq in self._db if eq.known_singular]

    def print_catalogue(self) -> None:
        print("═" * 68)
        print("  WHEELPHYSICS — Catalogue of physical equations")
        print("═" * 68)

        domains = {}
        for eq in self._db:
            domains.setdefault(eq.domain, []).append(eq)

        total_singular = sum(len(eq.known_singular) for eq in self._db)

        for domain, eqs in sorted(domains.items()):
            print(f"\n  [{domain}]")
            for eq in eqs:
                print(f"    {eq.one_liner()}")

        print(f"\n  Total: {len(self._db)} equations | {total_singular} known singularities")
        print("═" * 68)

    def stats(self) -> dict:
        return {
            "total":          len(self._db),
            "priority_1":     len(self.by_priority(Priority.DIVISION)),
            "priority_2":     len(self.by_priority(Priority.SINGULARITY)),
            "priority_3":     len(self.by_priority(Priority.LIMIT)),
            "total_singular": sum(len(eq.known_singular) for eq in self._db),
            "domains":        list({eq.domain for eq in self._db}),
        }


if __name__ == "__main__":
    db = EquationsDB()
    db.print_catalogue()
    print()
    stats = db.stats()
    print(f"  Statistics: {stats}")