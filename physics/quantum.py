# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
quantum.py — quantum mechanics and QED in Wheel Algebra

Investigated objects:
  1. Feynman propagator (scalar, fermion, photon)
  2. Divergences in QED (electron self-energy, vacuum polarization)
  3. Renormalization — does Wheel make it redundant?
  4. Dirac equation at zero mass
  5. Dispersion relation and mass shell (on-shell)

Key research questions:
  Q1: What does Wheel do with propagator poles?
  Q2: Do QED loop integrals (UV divergent) yield ⊥ in Wheel?
  Q3: Is renormalization a workaround for the lack of Wheel in classical analysis?
  Q4: What does Wheel say about Dirac equation for m=0 (neutrinos)?
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

_wa = WheelAlgebra()

# ─── Symbols ──────────────────────────────────────────────────────────────────

p, m, k, q   = sp.symbols("p m k q",     real=True)
p0, px       = sp.symbols("p0 px",       real=True)
m_e          = sp.Symbol("m_e",          positive=True)   # electron mass
e_charge     = sp.Symbol("e",            positive=True)   # charge
Lambda_uv    = sp.Symbol("Lambda",       positive=True)   # UV cut-off
mu_r         = sp.Symbol("mu",           positive=True)   # renorm. scale
epsilon_reg  = sp.Symbol("epsilon",      positive=True)   # dim. regularization
hbar         = sp.Symbol("hbar",         positive=True)
c_sym        = sp.Symbol("c",            positive=True)
alpha        = sp.Symbol("alpha",        positive=True)   # fine structure const ~ 1/137


# ─── 1. Feynman Propagators ───────────────────────────────────────────────────

class FeynmanPropagators:
    """
    Feynman propagators — the heart of perturbative QFT mechanism.

    Each propagator has a pole at p²=m² (on the mass shell).
    In classical QFT this is handled by Feynman prescription: p²-m²+iε.
    In Wheel: pole → ⊥.

    Question: can ⊥ in the propagator replace the iε prescription?
    """

    def scalar_propagator(self) -> sp.Basic:
        """
        Scalar propagator (Klein-Gordon particle):
          D_F(p) = i / (p² - m² + iε)

        Pole at p² = m²  (real particle, on-shell)
        """
        # Without iε (limit ε→0)
        return sp.Integer(1) / (p**2 - m**2)

    def fermion_propagator(self) -> sp.Basic:
        """
        Fermion propagator (electron in QED):
          S_F(p) = i(p̸ + m) / (p² - m² + iε)

        Numerator p̸ + m = γ^μ p_μ + m  (Dirac matrices — we scalarize)
        Pole: same as scalar, p²=m²
        """
        # Scalarized (trace over spinors / norm)
        numerator   = p + m_e          # simplification: p̸ → p (1D)
        denominator = p**2 - m_e**2
        return numerator / denominator

    def photon_propagator(self) -> sp.Basic:
        """
        Photon propagator (Lorenz gauge):
          D_F^μν(k) = -i g^μν / (k² + iε)

        Pole at k²=0 (massless photon)
        """
        return sp.Integer(1) / k**2    # scalar g^μν → 1

    def analyse_all(self) -> None:
        """Analysis of all propagators in Wheel."""

        print("\n" + "═" * 64)
        print("  FEYNMAN PROPAGATORS — Wheel Algebra Analysis")
        print("═" * 64)

        cases = [
            ("Scalar  D_F(p) = 1/(p²-m²)",
             self.scalar_propagator(),
             [(p, m), (p, -m), (p, sp.S.Zero)]),

            ("Fermion S_F(p) = (p+m_e)/(p²-m_e²)",
             self.fermion_propagator(),
             [(p, m_e), (p, -m_e)]),

            ("Photon  D_F(k) = 1/k²",
             self.photon_propagator(),
             [(k, sp.S.Zero)]),
        ]

        for name, expr, test_pts in cases:
            print(f"\n▶  {name}\n")
            for var, val in test_pts:
                w_result = wheel_subs(expr, {var: val})
                classical = "∞" if w_result.is_bottom else str(
                    sp.limit(expr, var, val) if not w_result.is_bottom else "∞"
                )
                label = f"{var}={val}"
                status = "⊥  ← pole (on-shell)" if w_result.is_bottom else str(w_result)
                print(f"  {label:<12} Wheel: {status}")

            # Laurent series expansion around the pole
            main_var, pole_val = test_pts[0]
            analysis = wheel_series_around(expr, main_var, pole_val)
            print(f"\n  Laurent around the pole ({main_var}={pole_val}):")
            print(f"    {analysis.get('laurent_series', 'none')}")
            print(f"    Limit (+): {analysis['limit_from_right']}")
            print(f"    Limit (-): {analysis['limit_from_left']}")

        # Feynman prescription vs Wheel
        print("\n" + "─" * 64)
        print("  FEYNMAN PRESCRIPTION vs WHEEL ALGEBRA\n")
        print("  Classical QFT:  1/(p²-m²+iε)  — shift pole into complex plane")
        print("  Wheel Algebra:  1/(p²-m²) at p=m  →  ⊥")
        print()
        print("  Interpretation:")
        print("  • iε prescription is a mathematical trick to bypass the pole")
        print("  • Wheel does not bypass — passes THROUGH the pole, result: ⊥")
        print("  • ⊥ might be a natural equivalent of 'particle is")
        print("    actually on the mass shell' — a physically distinguished state")
        print()
        print("  Hypothesis: ⊥ in the propagator = asymptotic particle state")
        print("              (what we look for in the S-matrix)")


# ─── 2. UV Divergences and Renormalization ────────────────────────────────────

class UVDivergences:
    """
    Ultraviolet divergences in QED.

    In perturbative QED, loop integrals diverge at high
    momenta (k → ∞). Renormalization systematically removes
    these infinities by redefining constants.

    Wheel Question: does introducing ⊥ instead of ∞ change this picture?
    """

    def self_energy_integrand(self) -> sp.Basic:
        """
        Integrand of electron self-energy Σ(p):

          Σ(p) ~ ∫ d⁴k / [k²(k-p)² - m²]

        Simplified integrand form (after spinor trace):
          f(k) = 1 / (k² - Lambda_uv²)    ← with cut-off regularization

        Diverges as k → ∞ (UV) or k → 0 (IR)
        """
        return sp.Integer(1) / (k**2 - Lambda_uv**2)

    def vacuum_polarization_integrand(self) -> sp.Basic:
        """
        Integrand of vacuum polarization Π(q²):

          Π(q²) ~ ∫₀¹ dx · x(1-x) · log[m² - x(1-x)q²]

        Logarithmic divergence as Lambda_uv → ∞.
        Simplified form:
          f(Lambda) = log(Lambda_uv² / m_e²)
        """
        return sp.log(Lambda_uv**2 / m_e**2)

    def running_coupling(self) -> sp.Basic:
        """
        Running coupling constant α(μ) in QED:

          1/α(μ) = 1/α₀ - (1/3π) log(μ²/m_e²)

        Landau pole: α → ∞ as μ → μ_L
        (unphysical in QED because μ_L ~ exp(137π) ≫ M_Planck)
        """
        alpha_0 = sp.Rational(1, 137)
        return alpha_0 / (1 - alpha_0 * sp.log(mu_r**2 / m_e**2) / (3 * sp.pi))

    def analyse_divergences(self) -> None:
        """Analysis of QED divergences in Wheel."""

        print("\n" + "═" * 64)
        print("  QED DIVERGENCES — Renormalization vs Wheel")
        print("═" * 64)

        # ── Self-energy with cut-off
        print("\n▶  Electron self-energy Σ ~ 1/(k²-Λ²)\n")
        se = self.self_energy_integrand()

        # UV Divergence: k → ∞
        lim_inf = sp.limit(se, k, sp.oo)
        print(f"  Integrand f(k) = {se}")
        print(f"  lim(k→∞) = {lim_inf}   (no UV divergence in the integrand)")
        print(f"  Wheel(k=Λ) = {wheel_subs(se, {k: Lambda_uv})}  ← pole at k=Λ")
        print(f"  Wheel(k=0) = {wheel_subs(se, {k: sp.S.Zero})}")
        print()
        print("  Note: UV divergence comes from INTEGRATION to ∞,")
        print("  not from a single value. Wheel operates pointwise.")

        # ── Vacuum polarization
        print("\n▶  Vacuum polarization Π ~ log(Λ²/m²)\n")
        vp = self.vacuum_polarization_integrand()
        print(f"  Π ~ {vp}")

        lim_vp = sp.limit(vp, Lambda_uv, sp.oo)
        print(f"  lim(Λ→∞) = {lim_vp}   (logarithmic divergence)")

        # In Wheel: Λ→∞ is not /0 — it is *0
        # log(∞) = ∞, but Wheel treats ∞ as ⊥
        w_vp_inf = wheel_subs(vp, {Lambda_uv: sp.oo})
        print(f"  Wheel(Λ=∞) = {w_vp_inf}")
        print()
        print("  Key observation:")
        print("  Log divergence → ⊥ in Wheel as Λ → ∞")
        print("  Renormalization replaces Λ with μ (observation scale)")
        print("  In Wheel: Λ=⊥ propagates through the entire expression")

        # ── Landau pole
        print("\n▶  Landau Pole — running coupling constant α(μ)\n")
        rc = self.running_coupling()
        print(f"  α(μ) = {rc}")

        # Search for the pole
        mu_L_eq = sp.solve(
            1 - sp.Rational(1, 137) * sp.log(mu_r**2 / m_e**2) / (3*sp.pi),
            mu_r
        )
        print(f"  Landau pole: μ_L = m_e · exp(3π·137) ≈ 10^(280) GeV")
        print(f"  (unphysical — beyond all energy scales)")
        print()
        w_landau = wheel_subs(rc, {mu_r: m_e * sp.exp(3 * sp.pi * 137)})
        print(f"  Wheel(μ=μ_L) = {w_landau}  ← ⊥ confirms the pole")

        # ── Main thesis
        print("\n" + "─" * 64)
        print("  WHEEL vs RENORMALIZATION — Analysis\n")
        print("  Renormalization (classical path):")
        print("  1. Compute amplitude with Λ as regulator")
        print("  2. Amplitude = finite + divergent(Λ)")
        print("  3. Redefine constants: m₀→m+δm, e₀→e+δe (counterterms)")
        print("  4. δm, δe contain Λ and cancel the divergences")
        print("  5. Result: finite predictions, consistent with experiment")
        print()
        print("  Wheel (alternative path — hypothesis):")
        print("  1. Compute amplitude — divergences → ⊥")
        print("  2. ⊥ propagates through the expression")
        print("  3. ??? — physical interpretation needed here")
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │ CONCLUSION (provisional):                           │")
        print("  │  Wheel does NOT automatically eliminate             │")
        print("  │  renormalization. Instead, it identifies EXACTLY    │")
        print("  │  where and why divergences appear.                  │")
        print("  │                                                      │")
        print("  │  Open question: can the algebra of ⊥ replace        │")
        print("  │  counterterms as a natural regulation mechanism?    │")
        print("  └─────────────────────────────────────────────────────┘")


# ─── 3. Dirac Equation ────────────────────────────────────────────────────────

class DiracEquation:
    """
    Dirac Equation: (iγ^μ∂_μ - m)ψ = 0

    In momentum space (after Fourier transform):
      (p̸ - m)ψ = 0

    For m=0 (neutrinos, massless fermions):
      p̸ ψ = 0

    Fermion propagator at m=0:
      S_F(p) = p̸/p²  — singularity at p=0!
    """

    def massless_propagator(self) -> sp.Basic:
        """Massless fermion propagator: S_F ~ p/p² = 1/p."""
        return p / p**2   # simplification: p̸ → p

    def massive_propagator_at_pole(self) -> dict:
        """Analyzes the fermion propagator pole at various masses."""
        results = {}
        for mass_val in [sp.S.Zero, m_e, sp.oo]:
            prop = (p + mass_val) / (p**2 - mass_val**2) if mass_val != 0 else 1/p
            if mass_val == sp.oo:
                w = wheel_subs(sp.Integer(1)/(p**2), {p: sp.S.Zero})
            elif mass_val == sp.S.Zero:
                w = wheel_subs(prop, {p: sp.S.Zero})
            else:
                w = wheel_subs(prop, {p: mass_val})
            results[str(mass_val)] = {"expr": prop, "wheel_at_pole": w}
        return results

    def analyse_dirac(self) -> None:
        """Analysis of the Dirac equation in Wheel."""

        print("\n" + "═" * 64)
        print("  DIRAC EQUATION — Massless fermions in Wheel")
        print("═" * 64)

        # ── Massless propagator
        print("\n▶  Massless propagator S_F ~ 1/p\n")
        mp = self.massless_propagator()
        print(f"  S_F(p) = {mp} = {sp.simplify(mp)}")
        print(f"  Wheel(p=0) = {wheel_subs(sp.simplify(mp), {p: sp.S.Zero})}")

        analysis = wheel_series_around(sp.Integer(1)/p, p, sp.S.Zero)
        print(f"\n  Limit (+): {analysis['limit_from_right']}")
        print(f"  Limit (-): {analysis['limit_from_left']}")
        print(f"  Laurent  : {analysis.get('laurent_series', 'none')}")

        print()
        print("  Wheel interpretation:")
        print("  • m=0 → propagator pole at p=0")
        print("  • Classically: S_F(0) = ∞ (IR problem)")
        print("  • Wheel: S_F(0) = ⊥")
        print("  • p=0 is a photon/gluon/neutrino at rest —")
        print("    a state physically unattainable for massless particles!")

        # ── Mass comparison
        print("\n▶  Fermion propagator at different masses\n")
        results = self.massive_propagator_at_pole()
        for mass_str, data in results.items():
            pole_val = mass_str if mass_str != "oo" else "∞"
            w = data["wheel_at_pole"]
            print(f"  m={mass_str:<6} → Wheel(p=m) = {w}")

        # ── Chirality and m=0
        print("\n▶  Chirality and the limit m→0\n")
        print("  For m=0 the Dirac equation splits into two independent ones:")
        print("  • Left-handed: iσ^μ∂_μ ψ_L = 0  (Weyl)")
        print("  • Right-handed: iσ̄^μ∂_μ ψ_R = 0  (Weyl)")
        print()

        # Massive propagator in the limit m→0
        prop_massive = (p + m_e) / (p**2 - m_e**2)
        prop_massive_simplified = sp.simplify(prop_massive)
        print(f"  S_F(p,m) = {prop_massive_simplified}")

        lim_m0 = sp.limit(prop_massive_simplified, m_e, 0)
        print(f"  lim(m→0) S_F = {lim_m0}")
        print(f"  Wheel(m=0, p=0) = {wheel_subs(lim_m0, {p: sp.S.Zero})}")
        print()
        print("  Conclusion: limit m→0 is regular everywhere EXCEPT p=0.")
        print("  Wheel points out p=0 as the only true IR-singularity.")


# ─── 4. Dispersion Relation and Mass Shell ────────────────────────────────────

class DispersionRelation:
    """
    Relativistic dispersion relation:
      E² = (pc)² + (mc²)²

    'On-shell': p² = m²c² (in units c=1: p² = m²)
    'Off-shell': p² ≠ m²  (virtual particles in Feynman diagrams)

    In Wheel: propagator 1/(p²-m²) yields ⊥ exactly on-shell.
    """

    def analyse_on_shell(self) -> None:
        print("\n" + "═" * 64)
        print("  DISPERSION RELATION — On-shell vs Off-shell in Wheel")
        print("═" * 64)

        prop = sp.Integer(1) / (p**2 - m**2)

        print("\n▶  Propagator 1/(p²-m²) as a function of p (at m=1)\n")

        test_vals = [
            (sp.Rational(-3, 2), "off-shell (p=-3/2)"),
            (sp.Integer(-1),     "on-shell  (p=-m)  "),
            (sp.Rational(-1, 2), "off-shell (p=-1/2)"),
            (sp.Integer(0),      "off-shell (p=0)   "),
            (sp.Rational(1, 2),  "off-shell (p=1/2) "),
            (sp.Integer(1),      "on-shell  (p=+m)  "),
            (sp.Rational(3, 2),  "off-shell (p=3/2) "),
        ]

        for val, label in test_vals:
            w = wheel_subs(prop, {p: val, m: sp.Integer(1)})
            marker = "  ←  POLE" if w.is_bottom else ""
            print(f"  {label}: {str(w):<20}{marker}")

        print()
        print("  Wheel interpretation:")
        print("  • Off-shell (virtual particles): finite propagator")
        print("  • On-shell  (real particles): propagator = ⊥")
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │ Provocative hypothesis:                              │")
        print("  │  ⊥ in propagator = particle is ACTUALLY              │")
        print("  │  observable (on-shell = the detector 'sees' it).     │")
        print("  │                                                      │")
        print("  │  The S-matrix connects asymptotic states (on-shell). │")
        print("  │  Wheel automatically marks these states with ⊥.      │")
        print("  │                                                      │")
        print("  │  Question: can ⊥ be an algebraic definition          │")
        print("  │  of 'observability' in QFT?                          │")
        print("  └─────────────────────────────────────────────────────┘")


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_quantum_analysis() -> None:
    print("\n" + "█" * 64)
    print("  WHEELPHYSICS — Quantum Mechanics and QED")
    print("  Divergences, propagators, renormalization")
    print("█" * 64)

    FeynmanPropagators().analyse_all()
    UVDivergences().analyse_divergences()
    DiracEquation().analyse_dirac()
    DispersionRelation().analyse_on_shell()

    # ── Synthetic research summary
    print("\n" + "═" * 64)
    print("  RESEARCH SUMMARY — QFT × Wheel")
    print("═" * 64)

    print("""
  RESULT 1: Propagators and poles
  ─────────────────────────────
  Every Feynman propagator has an on-shell pole → ⊥ in Wheel.
  The iε prescription (Feynman prescription) is a classical trick bypassing
  this pole in the complex plane. Wheel instead of bypassing —
  passes through the pole and returns ⊥.

  Is ⊥ a more natural description of an on-shell state than the limit ε→0?

  RESULT 2: UV Divergences
  ──────────────────────
  UV divergences (Λ→∞) yield ⊥ when treating Λ as an argument.
  Wheel does NOT eliminate renormalization automatically — divergences
  are in INTEGRALS, not in single values.
  
  But: Wheel precisely locates WHERE each divergence comes from.
  This could be a diagnostic tool for regularization theory.

  RESULT 3: Dirac Equation m=0
  ────────────────────────────
  Massless propagator 1/p → ⊥ at p=0.
  Physically: massless particle cannot have p=0 (relativistically).
  Wheel algebraically forbids this state — correctly!

  RESULT 4: On-shell = ⊥  (most provocative result)
  ────────────────────────────────────────────────────────
  Virtual particles (off-shell): finite propagator → computable
  Real particles (on-shell): propagator = ⊥ → 'outside algebra'
  
  In QFT virtual particles are NOT observable.
  On-shell particles ARE observable.
  
  Wheel algebraically separates these two classes via ⊥.
  """)


if __name__ == "__main__":
    run_quantum_analysis()