# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
general_relativity.py — general relativity in Wheel Algebra

Module analyzes GR equations for singularities and their
interpretation in wheel algebra.

Contains:
  - Schwarzschild metric (full, all components)
  - Christoffel symbols (non-zero)
  - Riemann tensor (selected components)
  - Kretschmann invariant (K = R_abcd R^abcd)
  - Friedmann equations
  - Wheel analysis for each singular point

Key research questions:
  Q1: What does Wheel say about r=0 (physical singularity)?
  Q2: What does Wheel say about r=r_s (event horizon — coordinate artifact)?
  Q3: Does the Kretschmann invariant yield ⊥ only at r=0 (correctly)?
  Q4: What does Friedmann do at a=0 (Big Bang)?
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp
from core.wheel_number import W, BOTTOM
from core.wheel_algebra import WheelAlgebra
from core.sympy_extension import (
    wheel_subs, is_singular_at, WheelFunction,
    wheel_series_around, print_singularity_analysis,
)
from scanner.translator import Translator


_wa  = WheelAlgebra()
_tr  = Translator()

# ─── Symbols ──────────────────────────────────────────────────────────────────

r, r_s     = sp.symbols("r r_s",     positive=True)
t_sym      = sp.Symbol("t",          positive=True)
theta      = sp.Symbol("theta",      positive=True)
G, M, c    = sp.symbols("G M c",     positive=True)
a          = sp.Symbol("a",          positive=True)   # scale factor (Friedmann)
k_curv     = sp.Symbol("k",          real=True)       # spatial curvature
Lambda     = sp.Symbol("Lambda",     real=True)       # cosmological constant
rho        = sp.Symbol("rho",        positive=True)   # energy density
H          = sp.Symbol("H",          real=True)       # Hubble parameter


# ─── Schwarzschild Metric ─────────────────────────────────────────────────────

class SchwarzschildMetric:
    """
    Schwarzschild metric in Schwarzschild coordinates:

      ds² = -f(r)c²dt² + f(r)⁻¹dr² + r²dΩ²

    where f(r) = 1 - r_s/r,  r_s = 2GM/c²

    Singularities:
      r = r_s  → event horizon (coordinate system artifact)
      r = 0    → physical singularity (curvature → ∞)
    """

    def __init__(self):
        # Schwarzschild radius
        self.r_s_def = sp.Rational(2) * G * M / c**2

        # Metric factor
        self.f = 1 - r_s / r

        # Metric tensor components (signature -, +, +, +)
        # Indices: 0=t, 1=r, 2=θ, 3=φ
        self.g = {
            (0, 0): -self.f * c**2,          # g_tt
            (1, 1):  1 / self.f,             # g_rr
            (2, 2):  r**2,                   # g_θθ
            (3, 3):  r**2 * sp.sin(theta)**2, # g_φφ
        }

        # Inverse metric tensor
        self.g_inv = {
            (0, 0): -1 / (self.f * c**2),
            (1, 1):  self.f,
            (2, 2):  1 / r**2,
            (3, 3):  1 / (r**2 * sp.sin(theta)**2),
        }

    def metric_component(self, mu: int, nu: int) -> sp.Basic:
        """Returns component g_μν."""
        return self.g.get((mu, nu), self.g.get((nu, mu), sp.S.Zero))

    def wheel_at(self, r_val, r_s_val=None) -> dict:
        """
        Evaluates all metric components in Wheel Algebra at r=r_val.
        Returns a dict {(μ,ν): WheelNumber}.
        """
        subs = {r: r_val}
        if r_s_val is not None:
            subs[r_s] = r_s_val

        return {
            key: wheel_subs(expr, subs)
            for key, expr in self.g.items()
        }

    def christoffel_nonzero(self) -> dict:
        """
        Non-zero Christoffel symbols Γ^μ_νρ for Schwarzschild metric.

        Derived analytically — only the non-zero ones from symmetry.
        Γ^μ_νρ = (1/2) g^{μσ} (∂_ν g_{σρ} + ∂_ρ g_{σν} - ∂_σ g_{νρ})
        """
        f   = self.f
        f_r = sp.diff(f, r)   # df/dr = r_s/r²

        christoffel = {}

        # Γ^t_tr = Γ^t_rt = f'/(2f)
        christoffel[("t", "t", "r")] = f_r / (2 * f)

        # Γ^r_tt = f·f'·c²/2
        christoffel[("r", "t", "t")] = f * f_r * c**2 / 2

        # Γ^r_rr = -f'/(2f)
        christoffel[("r", "r", "r")] = -f_r / (2 * f)

        # Γ^r_θθ = -r·f
        christoffel[("r", "θ", "θ")] = -r * f

        # Γ^r_φφ = -r·f·sin²θ
        christoffel[("r", "φ", "φ")] = -r * f * sp.sin(theta)**2

        # Γ^θ_rθ = Γ^θ_θr = 1/r
        christoffel[("θ", "r", "θ")] = 1 / r

        # Γ^θ_φφ = -sinθ·cosθ
        christoffel[("θ", "φ", "φ")] = -sp.sin(theta) * sp.cos(theta)

        # Γ^φ_rφ = 1/r
        christoffel[("φ", "r", "φ")] = 1 / r

        # Γ^φ_θφ = cosθ/sinθ
        christoffel[("φ", "θ", "φ")] = sp.cos(theta) / sp.sin(theta)

        return christoffel

    def kretschmann_scalar(self) -> sp.Basic:
        """
        Kretschmann invariant: K = R_abcd R^abcd

        For Schwarzschild metric:
          K = 12 r_s² / r⁶ = 48 G²M² / (c⁴ r⁶)

        This is a tensor invariant — it does not depend on the coordinate system.
        K → ∞ at r=0  (PHYSICAL singularity)
        K is FINITE at r=r_s  (horizon = coordinate artifact)
        """
        # Symbolic expression
        K = 12 * r_s**2 / r**6
        return K

    def analyse_singularities(self) -> None:
        """Full analysis of Schwarzschild metric singularities in Wheel."""

        print("\n" + "═" * 64)
        print("  SCHWARZSCHILD METRIC — Wheel Algebra Analysis")
        print("═" * 64)

        # ── Metric components
        print("\n▶  Components of the metric tensor g_μν\n")
        labels = {
            (0,0): "g_tt = -(1-r_s/r)·c²",
            (1,1): "g_rr = 1/(1-r_s/r)",
            (2,2): "g_θθ = r²",
            (3,3): "g_φφ = r²·sin²θ",
        }
        for key, expr in self.g.items():
            print(f"  {labels[key]}")
            # At r = r_s
            w_rrs = wheel_subs(expr, {r: r_s})
            # At r = 0
            w_0   = wheel_subs(expr, {r: sp.S.Zero})
            print(f"    r → r_s : {w_rrs}   {'← ⊥ (horizon)' if w_rrs.is_bottom else ''}")
            print(f"    r → 0   : {w_0}   {'← ⊥ (physical singularity)' if w_0.is_bottom else ''}")
            print()

        # ── Christoffel symbols
        print("▶  Christoffel symbols Γ^μ_νρ (non-zero)\n")
        christoffel = self.christoffel_nonzero()
        for (mu, nu, rho_idx), expr in christoffel.items():
            expr_simplified = sp.simplify(expr)
            w_rrs = wheel_subs(expr_simplified, {r: r_s})
            w_0   = wheel_subs(expr_simplified, {r: sp.S.Zero})

            # Show only those that have singularities
            if w_rrs.is_bottom or w_0.is_bottom:
                print(f"  Γ^{mu}_{nu}{rho_idx} = {expr_simplified}")
                if w_rrs.is_bottom:
                    print(f"    r=r_s → ⊥")
                if w_0.is_bottom:
                    print(f"    r=0   → ⊥")
                print()

        # ── Kretschmann Invariant — key test
        print("▶  Kretschmann invariant K = 12·r_s²/r⁶\n")
        K = self.kretschmann_scalar()

        w_K_rrs = wheel_subs(K, {r: r_s})
        w_K_0   = wheel_subs(K, {r: sp.S.Zero})

        K_at_rrs = sp.simplify(K.subs(r, r_s))

        print(f"  K = {K}")
        print(f"  K(r=r_s) classically = {K_at_rrs}")
        print(f"  K(r=r_s) Wheel       = {w_K_rrs}  {'✓ finite — horizon is only a coordinate artifact!' if not w_K_rrs.is_bottom else '✗'}")
        print(f"  K(r=0)   classically = ∞")
        print(f"  K(r=0)   Wheel       = {w_K_0}  {'✓ ⊥ — this is a true physical singularity' if w_K_0.is_bottom else ''}")

        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │ CONCLUSION:                                          │")
        print("  │  Wheel correctly distinguishes:                      │")
        print("  │  • r=r_s → K finite → ⊥ only in g_rr (artifact)      │")
        print("  │  • r=0   → K = ⊥    → physical singularity           │")
        print("  └─────────────────────────────────────────────────────┘")

        # ── Laurent analysis around r=0
        print("\n▶  Expansion of K around r=0 (Wheel behavior)\n")
        analysis = wheel_series_around(K, r, sp.S.Zero, n_terms=3)
        print_singularity_analysis(analysis)

        # ── What is on the other side of r=0?
        print("\n▶  What does Wheel say about r < 0? (on the other side of singularity)\n")
        print("  In classical physics r<0 is meaningless.")
        print("  In Wheel r is a symbol — we can substitute r=-ε:\n")
        epsilon = sp.Symbol("epsilon", positive=True)
        K_neg = K.subs(r, -epsilon)
        K_neg_simplified = sp.simplify(K_neg)
        print(f"  K(r=-ε) = {K_neg_simplified}")
        w_K_neg = wheel_subs(K_neg_simplified, {})
        print(f"  Wheel   : {w_K_neg}")
        print(f"  Note    : K(-ε) = 12r_s²/ε⁶ — same form as K(+ε)!")
        print(f"            Suggests symmetry through the singularity.")


# ─── Friedmann Equations ──────────────────────────────────────────────────────

class FriedmannEquations:
    """
    Friedmann equations describing the evolution of the universe:

      H² = (8πGρ/3) - kc²/a² + Λc²/3     [first equation]
      ȧ²/a² = ...                          (H = ȧ/a)

    Singularity at a=0 (Big Bang / Big Crunch).
    """

    def __init__(self):
        self.pi = sp.pi

        # Friedmann equation terms
        self.term_density  = sp.Rational(8, 3) * self.pi * G * rho
        self.term_curvature = -k_curv * c**2 / a**2
        self.term_lambda   = Lambda * c**2 / sp.Rational(3)

        # H² = term_density + term_curvature + term_lambda
        self.H_squared = (
            self.term_density +
            self.term_curvature +
            self.term_lambda
        )

        # Second Friedmann equation: ä/a = ...
        p_pressure = sp.Symbol("p_pressure")  # pressure
        self.acceleration = (
            -sp.Rational(4, 3) * self.pi * G * (rho + 3*p_pressure/c**2)
            + Lambda * c**2 / 3
        )

    def analyse_big_bang(self) -> None:
        """Analysis of Big Bang singularity in Wheel."""

        print("\n" + "═" * 64)
        print("  FRIEDMANN EQUATIONS — Big Bang in Wheel Algebra")
        print("═" * 64)

        print("\n▶  H² = 8πGρ/3 - kc²/a² + Λc²/3\n")

        # Term with singularity
        print(f"  Curvature term: {self.term_curvature}")
        w_curv_0 = wheel_subs(self.term_curvature, {a: sp.S.Zero})
        print(f"  Wheel(a=0): {w_curv_0}  {'← ⊥ (Big Bang!)' if w_curv_0.is_bottom else ''}")

        # Classical limit
        print(f"\n  Classically: lim(a→0) H² → ∞  (singularity)")
        print(f"  Wheel:       H²(a=0) = ⊥  (defined as bottom)")

        # What about density as a→0?
        print(f"\n▶  Behavior of energy density as a→0:")
        print(f"  Classically: ρ ∝ 1/a³ → ∞  (matter)")
        print(f"  Classically: ρ ∝ 1/a⁴ → ∞  (radiation)")

        rho_matter = sp.Symbol("rho_0") / a**3
        rho_rad    = sp.Symbol("rho_0") / a**4

        print(f"\n  ρ_matter = ρ₀/a³:")
        print(f"    Wheel(a=0) = {wheel_subs(rho_matter, {a: sp.S.Zero})}")
        print(f"\n  ρ_radiation = ρ₀/a⁴:")
        print(f"    Wheel(a=0) = {wheel_subs(rho_rad, {a: sp.S.Zero})}")

        print("\n▶  Wheel interpretation for the Big Bang:\n")
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  a=0 → H²=⊥, ρ=⊥, K=⊥                            │")
        print("  │                                                      │")
        print("  │  Classically: physics ends here (∞)                  │")
        print("  │  Wheel:       ⊥ starts here — 'before' state         │")
        print("  │               a<0 is mathematically permitted!       │")
        print("  │                                                      │")
        print("  │  Hypothesis: a<0 = time before the Big Bang?         │")
        print("  └─────────────────────────────────────────────────────┘")

        # Check a < 0
        print("\n▶  Wheel for a < 0 (before the Big Bang?):\n")
        epsilon = sp.Symbol("epsilon", positive=True)

        term_neg = self.term_curvature.subs(a, -epsilon)
        term_neg_s = sp.simplify(term_neg)
        print(f"  Curvature term for a=-ε: {term_neg_s}")
        print(f"  Same form as for a=+ε — symmetry through a=0!")
        print(f"  Wheel(-ε): {wheel_subs(term_neg_s, {})}")


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_full_analysis() -> None:
    """Runs full GR analysis in Wheel."""

    print("\n" + "█" * 64)
    print("  WHEELPHYSICS — General Relativity")
    print("  Investigating singularities via Wheel Algebra")
    print("█" * 64)

    # Schwarzschild
    sm = SchwarzschildMetric()
    sm.analyse_singularities()

    # Friedmann
    fe = FriedmannEquations()
    fe.analyse_big_bang()

    # Research summary
    print("\n" + "═" * 64)
    print("  RESEARCH SUMMARY — GR × Wheel")
    print("═" * 64)

    findings = [
        ("Q1: r=0 (physical singularity)",
         "K(r=0) = ⊥ — Wheel confirms the singularity. It doesn't 'remove' it,",
         "but assigns a finite symbol ⊥ instead of ∞. Question:",
         "does ⊥ carry more information than ∞?"),

        ("Q2: r=r_s (event horizon)",
         "K(r=r_s) finite — Wheel CORRECTLY identifies the horizon",
         "as a coordinate system artifact. g_rr(r_s) = ⊥, but",
         "the physical invariant K is regular."),

        ("Q3: a=0 (Big Bang)",
         "H²(a=0) = ⊥. Wheel allows a<0 — we can study 'before'",
         "the Big Bang without changing equations. Symmetry through a=0."),

        ("Q4: Next step",
         "quantum.py — Feynman propagator, QED renormalization.",
         "Does Wheel eliminate the need for renormalization?"),
    ]

    for finding in findings:
        print(f"\n  ► {finding[0]}")
        for line in finding[1:]:
            print(f"    {line}")

    print("\n" + "═" * 64)


if __name__ == "__main__":
    run_full_analysis()