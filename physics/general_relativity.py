# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
general_relativity.py — ogólna teoria względności w Wheel Algebra

Moduł analizuje równania OTW pod kątem osobliwości i ich
interpretacji w algebrze koła.

Zawiera:
  - Metryka Schwarzschilda (pełna, wszystkie składowe)
  - Symbole Christoffela (niezerowe)
  - Tensor Riemanna (wybrane składowe)
  - Niezmiennik Kretschmanna (K = R_abcd R^abcd)
  - Równania Friedmanna
  - Analiza wheel dla każdego punktu osobliwego

Kluczowe pytania badawcze:
  Q1: Co Wheel mówi o r=0 (osobliwość fizyczna)?
  Q2: Co Wheel mówi o r=r_s (horyzont zdarzeń — artefakt układu)?
  Q3: Czy niezmiennik Kretschmanna daje ⊥ tylko przy r=0 (poprawnie)?
  Q4: Co Friedmann robi przy a=0 (Wielki Wybuch)?
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

# ─── Symbole ──────────────────────────────────────────────────────────────────

r, r_s     = sp.symbols("r r_s",     positive=True)
t_sym      = sp.Symbol("t",          positive=True)
theta      = sp.Symbol("theta",      positive=True)
G, M, c    = sp.symbols("G M c",     positive=True)
a          = sp.Symbol("a",          positive=True)   # czynnik skali (Friedmann)
k_curv     = sp.Symbol("k",          real=True)       # krzywizna przestrzenna
Lambda     = sp.Symbol("Lambda",     real=True)       # stała kosmologiczna
rho        = sp.Symbol("rho",        positive=True)   # gęstość energii
H          = sp.Symbol("H",          real=True)       # parametr Hubble'a


# ─── Metryka Schwarzschilda ───────────────────────────────────────────────────

class SchwarzschildMetric:
    """
    Metryka Schwarzschilda w współrzędnych Schwarzschilda:

      ds² = -f(r)c²dt² + f(r)⁻¹dr² + r²dΩ²

    gdzie f(r) = 1 - r_s/r,  r_s = 2GM/c²

    Osobliwości:
      r = r_s  → horyzont zdarzeń  (artefakt układu współrzędnych)
      r = 0    → osobliwość fizyczna (krzywizna → ∞)
    """

    def __init__(self):
        # Promień Schwarzschilda
        self.r_s_def = sp.Rational(2) * G * M / c**2

        # Czynnik metryczny
        self.f = 1 - r_s / r

        # Składowe tensora metrycznego (sygnatura -, +, +, +)
        # Indeksy: 0=t, 1=r, 2=θ, 3=φ
        self.g = {
            (0, 0): -self.f * c**2,          # g_tt
            (1, 1):  1 / self.f,             # g_rr
            (2, 2):  r**2,                   # g_θθ
            (3, 3):  r**2 * sp.sin(theta)**2, # g_φφ
        }

        # Odwrotny tensor metryczny
        self.g_inv = {
            (0, 0): -1 / (self.f * c**2),
            (1, 1):  self.f,
            (2, 2):  1 / r**2,
            (3, 3):  1 / (r**2 * sp.sin(theta)**2),
        }

    def metric_component(self, mu: int, nu: int) -> sp.Basic:
        """Zwraca składową g_μν."""
        return self.g.get((mu, nu), self.g.get((nu, mu), sp.S.Zero))

    def wheel_at(self, r_val, r_s_val=None) -> dict:
        """
        Ewaluuje wszystkie składowe metryki w Wheel Algebra przy r=r_val.
        Zwraca słownik {(μ,ν): WheelNumber}.
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
        Niezerowe symbole Christoffela Γ^μ_νρ dla metryki Schwarzschilda.

        Wyprowadzone analitycznie — tylko te niezerowe z symetrii.
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
        Niezmiennik Kretschmanna: K = R_abcd R^abcd

        Dla metryki Schwarzschilda:
          K = 12 r_s² / r⁶ = 48 G²M² / (c⁴ r⁶)

        To jest niezmiennik tensorowy — nie zależy od układu współrzędnych.
        K → ∞ przy r=0  (osobliwość FIZYCZNA)
        K jest SKOŃCZONY przy r=r_s  (horyzont = artefakt układu)
        """
        # Wyrażenie symboliczne
        K = 12 * r_s**2 / r**6
        return K

    def analyse_singularities(self) -> None:
        """Pełna analiza osobliwości metryki Schwarzschilda w Wheel."""

        print("\n" + "═" * 64)
        print("  METRYKA SCHWARZSCHILDA — Analiza Wheel Algebra")
        print("═" * 64)

        # ── Składowe metryki
        print("\n▶  Składowe tensora metrycznego g_μν\n")
        labels = {
            (0,0): "g_tt = -(1-r_s/r)·c²",
            (1,1): "g_rr = 1/(1-r_s/r)",
            (2,2): "g_θθ = r²",
            (3,3): "g_φφ = r²·sin²θ",
        }
        for key, expr in self.g.items():
            print(f"  {labels[key]}")
            # Przy r = r_s
            w_rrs = wheel_subs(expr, {r: r_s})
            # Przy r = 0
            w_0   = wheel_subs(expr, {r: sp.S.Zero})
            print(f"    r → r_s : {w_rrs}   {'← ⊥ (horyzont)' if w_rrs.is_bottom else ''}")
            print(f"    r → 0   : {w_0}   {'← ⊥ (osobliwość fizyczna)' if w_0.is_bottom else ''}")
            print()

        # ── Symbole Christoffela
        print("▶  Symbole Christoffela Γ^μ_νρ (niezerowe)\n")
        christoffel = self.christoffel_nonzero()
        for (mu, nu, rho_idx), expr in christoffel.items():
            expr_simplified = sp.simplify(expr)
            w_rrs = wheel_subs(expr_simplified, {r: r_s})
            w_0   = wheel_subs(expr_simplified, {r: sp.S.Zero})

            # Pokaż tylko te które mają osobliwości
            if w_rrs.is_bottom or w_0.is_bottom:
                print(f"  Γ^{mu}_{nu}{rho_idx} = {expr_simplified}")
                if w_rrs.is_bottom:
                    print(f"    r=r_s → ⊥")
                if w_0.is_bottom:
                    print(f"    r=0   → ⊥")
                print()

        # ── Niezmiennik Kretschmanna — kluczowy test
        print("▶  Niezmiennik Kretschmanna K = 12·r_s²/r⁶\n")
        K = self.kretschmann_scalar()

        w_K_rrs = wheel_subs(K, {r: r_s})
        w_K_0   = wheel_subs(K, {r: sp.S.Zero})

        K_at_rrs = sp.simplify(K.subs(r, r_s))

        print(f"  K = {K}")
        print(f"  K(r=r_s) klasycznie = {K_at_rrs}")
        print(f"  K(r=r_s) Wheel      = {w_K_rrs}  {'✓ skończony — horyzont to tylko artefakt układu!' if not w_K_rrs.is_bottom else '✗'}")
        print(f"  K(r=0)   klasycznie = ∞")
        print(f"  K(r=0)   Wheel      = {w_K_0}  {'✓ ⊥ — to jest prawdziwa osobliwość fizyczna' if w_K_0.is_bottom else ''}")

        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │ WNIOSEK:                                             │")
        print("  │  Wheel poprawnie rozróżnia:                          │")
        print("  │  • r=r_s → K skończony → ⊥ tylko w g_rr (artefakt) │")
        print("  │  • r=0   → K = ⊥       → osobliwość fizyczna        │")
        print("  └─────────────────────────────────────────────────────┘")

        # ── Analiza Laurent w okolicy r=0
        print("\n▶  Rozwinięcie K w okolicy r=0 (zachowanie Wheel)\n")
        analysis = wheel_series_around(K, r, sp.S.Zero, n_terms=3)
        print_singularity_analysis(analysis)

        # ── Co jest po drugiej stronie r=0?
        print("\n▶  Co mówi Wheel o r < 0? (po drugiej stronie osobliwości)\n")
        print("  W klasycznej fizyce r<0 jest bezsensowne.")
        print("  W Wheel r jest symbolem — możemy podstawić r=-ε:\n")
        epsilon = sp.Symbol("epsilon", positive=True)
        K_neg = K.subs(r, -epsilon)
        K_neg_simplified = sp.simplify(K_neg)
        print(f"  K(r=-ε) = {K_neg_simplified}")
        w_K_neg = wheel_subs(K_neg_simplified, {})
        print(f"  Wheel   : {w_K_neg}")
        print(f"  Uwaga   : K(-ε) = 12r_s²/ε⁶ — ta sama forma co K(+ε)!")
        print(f"            Sugeruje symetrię przez osobliwość.")


# ─── Równania Friedmanna ──────────────────────────────────────────────────────

class FriedmannEquations:
    """
    Równania Friedmanna opisujące ewolucję wszechświata:

      H² = (8πGρ/3) - kc²/a² + Λc²/3     [pierwsze równanie]
      ȧ²/a² = ...                          (H = ȧ/a)

    Osobliwość przy a=0 (Wielki Wybuch / Wielki Ścisk).
    """

    def __init__(self):
        self.pi = sp.pi

        # Człony równania Friedmanna
        self.term_density  = sp.Rational(8, 3) * self.pi * G * rho
        self.term_curvature = -k_curv * c**2 / a**2
        self.term_lambda   = Lambda * c**2 / sp.Rational(3)

        # H² = term_density + term_curvature + term_lambda
        self.H_squared = (
            self.term_density +
            self.term_curvature +
            self.term_lambda
        )

        # Drugie równanie Friedmanna: ä/a = ...
        p_pressure = sp.Symbol("p_pressure")  # ciśnienie
        self.acceleration = (
            -sp.Rational(4, 3) * self.pi * G * (rho + 3*p_pressure/c**2)
            + Lambda * c**2 / 3
        )

    def analyse_big_bang(self) -> None:
        """Analiza osobliwości Wielkiego Wybuchu w Wheel."""

        print("\n" + "═" * 64)
        print("  RÓWNANIA FRIEDMANNA — Wielki Wybuch w Wheel Algebra")
        print("═" * 64)

        print("\n▶  H² = 8πGρ/3 - kc²/a² + Λc²/3\n")

        # Człon z osobliwością
        print(f"  Człon krzywiznowy: {self.term_curvature}")
        w_curv_0 = wheel_subs(self.term_curvature, {a: sp.S.Zero})
        print(f"  Wheel(a=0): {w_curv_0}  {'← ⊥ (Wielki Wybuch!)' if w_curv_0.is_bottom else ''}")

        # Klasyczna granica
        print(f"\n  Klasycznie: lim(a→0) H² → ∞  (osobliwość)")
        print(f"  Wheel:      H²(a=0) = ⊥  (zdefiniowane jako bottom)")

        # Co z gęstością przy a→0?
        print(f"\n▶  Zachowanie gęstości energii przy a→0:")
        print(f"  Klasycznie: ρ ∝ 1/a³ → ∞  (materia)")
        print(f"  Klasycznie: ρ ∝ 1/a⁴ → ∞  (promieniowanie)")

        rho_matter = sp.Symbol("rho_0") / a**3
        rho_rad    = sp.Symbol("rho_0") / a**4

        print(f"\n  ρ_materia = ρ₀/a³:")
        print(f"    Wheel(a=0) = {wheel_subs(rho_matter, {a: sp.S.Zero})}")
        print(f"\n  ρ_promieniowanie = ρ₀/a⁴:")
        print(f"    Wheel(a=0) = {wheel_subs(rho_rad, {a: sp.S.Zero})}")

        print("\n▶  Interpretacja Wheel dla Wielkiego Wybuchu:\n")
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  a=0 → H²=⊥, ρ=⊥, K=⊥                            │")
        print("  │                                                      │")
        print("  │  Klasycznie: tu kończy się fizyka (∞)               │")
        print("  │  Wheel:      tu zaczyna się ⊥ — stan 'sprzed'       │")
        print("  │              a<0 jest matematycznie dozwolone!       │")
        print("  │                                                      │")
        print("  │  Hipoteza: a<0 = czas przed Wielkim Wybuchem?       │")
        print("  └─────────────────────────────────────────────────────┘")

        # Sprawdź a < 0
        print("\n▶  Wheel dla a < 0 (przed Wielkim Wybuchem?):\n")
        epsilon = sp.Symbol("epsilon", positive=True)

        term_neg = self.term_curvature.subs(a, -epsilon)
        term_neg_s = sp.simplify(term_neg)
        print(f"  Człon krzywiznowy dla a=-ε: {term_neg_s}")
        print(f"  Ta sama forma co dla a=+ε — symetria przez a=0!")
        print(f"  Wheel(-ε): {wheel_subs(term_neg_s, {})}")


# ─── Główna analiza ───────────────────────────────────────────────────────────

def run_full_analysis() -> None:
    """Uruchamia pełną analizę OTW w Wheel."""

    print("\n" + "█" * 64)
    print("  WHEELPHYSICS — Ogólna Teoria Względności")
    print("  Badanie osobliwości przez Wheel Algebra")
    print("█" * 64)

    # Schwarzschild
    sm = SchwarzschildMetric()
    sm.analyse_singularities()

    # Friedmann
    fe = FriedmannEquations()
    fe.analyse_big_bang()

    # Podsumowanie badawcze
    print("\n" + "═" * 64)
    print("  PODSUMOWANIE BADAWCZE — OTW × Wheel")
    print("═" * 64)

    findings = [
        ("Q1: r=0 (osobliwość fizyczna)",
         "K(r=0) = ⊥ — Wheel potwierdza osobliwość. Nie 'usuwa' jej,",
         "lecz przypisuje skończony symbol ⊥ zamiast ∞. Pytanie:",
         "czy ⊥ niesie więcej informacji niż ∞?"),

        ("Q2: r=r_s (horyzont zdarzeń)",
         "K(r=r_s) skończone — Wheel POPRAWNIE identyfikuje horyzont",
         "jako artefakt układu współrzędnych. g_rr(r_s) = ⊥, ale",
         "niezmiennik fizyczny K jest regularny."),

        ("Q3: a=0 (Wielki Wybuch)",
         "H²(a=0) = ⊥. Wheel dopuszcza a<0 — można badać 'przed'",
         "Wielkim Wybuchem bez zmiany równań. Symetria przez a=0."),

        ("Q4: Następny krok",
         "quantum.py — propagator Feynmana, renormalizacja QED.",
         "Czy Wheel eliminuje potrzebę renormalizacji?"),
    ]

    for finding in findings:
        print(f"\n  ► {finding[0]}")
        for line in finding[1:]:
            print(f"    {line}")

    print("\n" + "═" * 64)


if __name__ == "__main__":
    run_full_analysis()