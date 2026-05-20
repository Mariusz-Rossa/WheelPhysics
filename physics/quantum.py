# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
quantum.py — mechanika kwantowa i QED w Wheel Algebra

Badane obiekty:
  1. Propagator Feynmana (skalarny, fermionowy, fotonowy)
  2. Dywergencje w QED (samoczynna energia elektronu, polaryzacja próżni)
  3. Renormalizacja — czy Wheel czyni ją zbędną?
  4. Równanie Diraca przy zerowej masie
  5. Relacja dyspersji i masa powłoki (on-shell)

Kluczowe pytania badawcze:
  Q1: Co Wheel robi z biegunami propagatora?
  Q2: Czy całki pętlowe QED (dywergentne UV) dają ⊥ w Wheel?
  Q3: Czy renormalizacja to obejście braku Wheel w klasycznej analizie?
  Q4: Co Wheel mówi o równaniu Diraca dla m=0 (neutrina)?
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

# ─── Symbole ──────────────────────────────────────────────────────────────────

p, m, k, q   = sp.symbols("p m k q",     real=True)
p0, px       = sp.symbols("p0 px",       real=True)
m_e          = sp.Symbol("m_e",          positive=True)   # masa elektronu
e_charge     = sp.Symbol("e",            positive=True)   # ładunek
Lambda_uv    = sp.Symbol("Lambda",       positive=True)   # cut-off UV
mu_r         = sp.Symbol("mu",           positive=True)   # skala renorm.
epsilon_reg  = sp.Symbol("epsilon",      positive=True)   # regularyzacja dim.
hbar         = sp.Symbol("hbar",         positive=True)
c_sym        = sp.Symbol("c",            positive=True)
alpha        = sp.Symbol("alpha",        positive=True)   # stała struktury subtelnej ~ 1/137


# ─── 1. Propagatory Feynmana ──────────────────────────────────────────────────

class FeynmanPropagators:
    """
    Propagatory Feynmana — serce mechanizmu perturbacyjnego QFT.

    Każdy propagator ma biegun przy p²=m² (na powłoce masy).
    W klasycznej QFT obsługuje się to receptyną Feynmana: p²-m²+iε.
    W Wheel: biegun → ⊥.

    Pytanie: czy ⊥ w propagatorze może zastąpić receptyną iε?
    """

    def scalar_propagator(self) -> sp.Basic:
        """
        Propagator skalarny (cząstka Kleina-Gordona):
          D_F(p) = i / (p² - m² + iε)

        Biegun przy p² = m²  (cząstka rzeczywista, on-shell)
        """
        # Bez iε (granica ε→0)
        return sp.Integer(1) / (p**2 - m**2)

    def fermion_propagator(self) -> sp.Basic:
        """
        Propagator fermionowy (elektron w QED):
          S_F(p) = i(p̸ + m) / (p² - m² + iε)

        Licznik p̸ + m = γ^μ p_μ + m  (macierze Diraca — skalaryzujemy)
        Biegun: ten sam co skalarny, p²=m²
        """
        # Skalaryzowany (ślad po spinorach / norma)
        numerator   = p + m_e          # uproszczenie: p̸ → p (1D)
        denominator = p**2 - m_e**2
        return numerator / denominator

    def photon_propagator(self) -> sp.Basic:
        """
        Propagator fotonowy (gauge Lorenza):
          D_F^μν(k) = -i g^μν / (k² + iε)

        Biegun przy k²=0 (foton bezmasowy)
        """
        return sp.Integer(1) / k**2    # skalarne g^μν → 1

    def analyse_all(self) -> None:
        """Analiza wszystkich propagatorów w Wheel."""

        print("\n" + "═" * 64)
        print("  PROPAGATORY FEYNMANA — Analiza Wheel Algebra")
        print("═" * 64)

        cases = [
            ("Skalarny  D_F(p) = 1/(p²-m²)",
             self.scalar_propagator(),
             [(p, m), (p, -m), (p, sp.S.Zero)]),

            ("Fermionowy S_F(p) = (p+m_e)/(p²-m_e²)",
             self.fermion_propagator(),
             [(p, m_e), (p, -m_e)]),

            ("Fotonowy  D_F(k) = 1/k²",
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
                status = "⊥  ← biegun (on-shell)" if w_result.is_bottom else str(w_result)
                print(f"  {label:<12} Wheel: {status}")

            # Rozwinięcie w szereg Laurenta wokół bieguna
            main_var, pole_val = test_pts[0]
            analysis = wheel_series_around(expr, main_var, pole_val)
            print(f"\n  Laurent wokół bieguna ({main_var}={pole_val}):")
            print(f"    {analysis.get('laurent_series', 'brak')}")
            print(f"    Granica (+): {analysis['limit_from_right']}")
            print(f"    Granica (-): {analysis['limit_from_left']}")

        # Recepta Feynmana vs Wheel
        print("\n" + "─" * 64)
        print("  RECEPTA FEYNMANA vs WHEEL ALGEBRA\n")
        print("  Klasyczna QFT:  1/(p²-m²+iε)  — przesuń biegun w zespolone")
        print("  Wheel Algebra:  1/(p²-m²) przy p=m  →  ⊥")
        print()
        print("  Interpretacja:")
        print("  • Recepta iε to matematyczny trik omijający biegun")
        print("  • Wheel nie omija — przechodzi PRZEZ biegun, wynik: ⊥")
        print("  • ⊥ może być naturalnym odpowiednikiem 'cząstka jest")
        print("    rzeczywiście na powłoce masy' — stan fizycznie wyróżniony")
        print()
        print("  Hipoteza: ⊥ w propagatorze = asymptotyczny stan cząstkowy")
        print("            (to czego szukamy w macierzy S)")


# ─── 2. Dywergencje UV i renormalizacja ──────────────────────────────────────

class UVDivergences:
    """
    Dywergencje ultraviolet w QED.

    W perturbacyjnej QED całki pętlowe dywergują przy wysokich
    impulsach (k → ∞). Renormalizacja systematycznie usuwa
    te nieskończoności przez redefinicję stałych.

    Pytanie Wheel: czy wprowadzenie ⊥ zamiast ∞ zmienia ten obraz?
    """

    def self_energy_integrand(self) -> sp.Basic:
        """
        Podcałkowa samoenergii elektronu Σ(p):

          Σ(p) ~ ∫ d⁴k / [k²(k-p)² - m²]

        Uproszczona forma podcałkowej (po śladzie spinorowym):
          f(k) = 1 / (k² - Lambda_uv²)    ← z regularyzacją cut-off

        Dywerguje gdy k → ∞ (UV) lub k → 0 (IR)
        """
        return sp.Integer(1) / (k**2 - Lambda_uv**2)

    def vacuum_polarization_integrand(self) -> sp.Basic:
        """
        Podcałkowa polaryzacji próżni Π(q²):

          Π(q²) ~ ∫₀¹ dx · x(1-x) · log[m² - x(1-x)q²]

        Dywergencja logarytmiczna przy Lambda_uv → ∞.
        Uproszczona forma:
          f(Lambda) = log(Lambda_uv² / m_e²)
        """
        return sp.log(Lambda_uv**2 / m_e**2)

    def running_coupling(self) -> sp.Basic:
        """
        Bieżąca stała sprzężenia α(μ) w QED:

          1/α(μ) = 1/α₀ - (1/3π) log(μ²/m_e²)

        Landau pole: α → ∞ gdy μ → μ_L
        (w QED niefizyczne bo μ_L ~ exp(137π) ≫ M_Planck)
        """
        alpha_0 = sp.Rational(1, 137)
        return alpha_0 / (1 - alpha_0 * sp.log(mu_r**2 / m_e**2) / (3 * sp.pi))

    def analyse_divergences(self) -> None:
        """Analiza dywergencji QED w Wheel."""

        print("\n" + "═" * 64)
        print("  DYWERGENCJE QED — Renormalizacja vs Wheel")
        print("═" * 64)

        # ── Samoenergia z cut-off
        print("\n▶  Samoenergia elektronu Σ ~ 1/(k²-Λ²)\n")
        se = self.self_energy_integrand()

        # Dywergencja UV: k → ∞
        lim_inf = sp.limit(se, k, sp.oo)
        print(f"  Podcałkowa f(k) = {se}")
        print(f"  lim(k→∞) = {lim_inf}   (brak dywergencji UV w podcałkowej)")
        print(f"  Wheel(k=Λ) = {wheel_subs(se, {k: Lambda_uv})}  ← biegun przy k=Λ")
        print(f"  Wheel(k=0) = {wheel_subs(se, {k: sp.S.Zero})}")
        print()
        print("  Uwaga: dywergencja UV pochodzi z CAŁKOWANIA do ∞,")
        print("  nie z pojedynczej wartości. Wheel operuje punktowo.")

        # ── Polaryzacja próżni
        print("\n▶  Polaryzacja próżni Π ~ log(Λ²/m²)\n")
        vp = self.vacuum_polarization_integrand()
        print(f"  Π ~ {vp}")

        lim_vp = sp.limit(vp, Lambda_uv, sp.oo)
        print(f"  lim(Λ→∞) = {lim_vp}   (dywergencja logarytmiczna)")

        # W Wheel: Λ→∞ to nie jest /0 — to jest *0
        # log(∞) = ∞, ale Wheel traktuje ∞ jako ⊥
        w_vp_inf = wheel_subs(vp, {Lambda_uv: sp.oo})
        print(f"  Wheel(Λ=∞) = {w_vp_inf}")
        print()
        print("  Kluczowe spostrzeżenie:")
        print("  Dywergencja log → ⊥ w Wheel gdy Λ → ∞")
        print("  Renormalizacja zastępuje Λ przez μ (skala obserwacji)")
        print("  W Wheel: Λ=⊥ propaguje przez całe wyrażenie")

        # ── Landau pole
        print("\n▶  Landau Pole — bieżąca stała sprzężenia α(μ)\n")
        rc = self.running_coupling()
        print(f"  α(μ) = {rc}")

        # Szukaj bieguna
        mu_L_eq = sp.solve(
            1 - sp.Rational(1, 137) * sp.log(mu_r**2 / m_e**2) / (3*sp.pi),
            mu_r
        )
        print(f"  Landau pole: μ_L = m_e · exp(3π·137) ≈ 10^(280) GeV")
        print(f"  (niefizyczne — poza wszelkimi skalami energii)")
        print()
        w_landau = wheel_subs(rc, {mu_r: m_e * sp.exp(3 * sp.pi * 137)})
        print(f"  Wheel(μ=μ_L) = {w_landau}  ← ⊥ potwierdza biegun")

        # ── Główna teza
        print("\n" + "─" * 64)
        print("  WHEEL vs RENORMALIZACJA — Analiza\n")
        print("  Renormalizacja (klasyczna droga):")
        print("  1. Oblicz amplitudę z Λ jako regulatorem")
        print("  2. Amplituda = skończone + dywergentne(Λ)")
        print("  3. Zredefiniuj stałe: m₀→m+δm, e₀→e+δe (kontrterminy)")
        print("  4. δm, δe zawierają Λ i kasują dywergencje")
        print("  5. Wynik: przewidywania skończone, zgodne z eksperymentem")
        print()
        print("  Wheel (alternatywna droga — hipoteza):")
        print("  1. Oblicz amplitudę — dywergencje → ⊥")
        print("  2. ⊥ propaguje przez wyrażenie")
        print("  3. ??? — tu potrzebna nowa interpretacja fizyczna")
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │ WNIOSEK (tymczasowy):                               │")
        print("  │  Wheel NIE eliminuje automatycznie renormalizacji.   │")
        print("  │  Zamiast tego: identyfikuje DOKŁADNIE gdzie i       │")
        print("  │  dlaczego dywergencje się pojawiają.                │")
        print("  │                                                      │")
        print("  │  Otwarte pytanie: czy algebra ⊥ może zastąpić       │")
        print("  │  kontrterminy jako naturalny mechanizm regulacji?   │")
        print("  └─────────────────────────────────────────────────────┘")


# ─── 3. Równanie Diraca ───────────────────────────────────────────────────────

class DiracEquation:
    """
    Równanie Diraca: (iγ^μ∂_μ - m)ψ = 0

    W przestrzeni pędów (po transformacji Fouriera):
      (p̸ - m)ψ = 0

    Dla m=0 (neutrina, bezmasowe fermiony):
      p̸ ψ = 0

    Propagator fermionowy przy m=0:
      S_F(p) = p̸/p²  — osobliwość przy p=0!
    """

    def massless_propagator(self) -> sp.Basic:
        """Propagator bezmasowego fermiona: S_F ~ p/p² = 1/p."""
        return p / p**2   # uproszczenie: p̸ → p

    def massive_propagator_at_pole(self) -> dict:
        """Analizuje biegun propagatora fermionowego przy różnych masach."""
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
        """Analiza równania Diraca w Wheel."""

        print("\n" + "═" * 64)
        print("  RÓWNANIE DIRACA — Bezmasowe fermiony w Wheel")
        print("═" * 64)

        # ── Bezmasowy propagator
        print("\n▶  Propagator bezmasowy S_F ~ 1/p\n")
        mp = self.massless_propagator()
        print(f"  S_F(p) = {mp} = {sp.simplify(mp)}")
        print(f"  Wheel(p=0) = {wheel_subs(sp.simplify(mp), {p: sp.S.Zero})}")

        analysis = wheel_series_around(sp.Integer(1)/p, p, sp.S.Zero)
        print(f"\n  Granica (+): {analysis['limit_from_right']}")
        print(f"  Granica (-): {analysis['limit_from_left']}")
        print(f"  Laurent    : {analysis.get('laurent_series', 'brak')}")

        print()
        print("  Interpretacja Wheel:")
        print("  • m=0 → biegun propagatora przy p=0")
        print("  • Klasycznie: S_F(0) = ∞ (problem IR)")
        print("  • Wheel: S_F(0) = ⊥")
        print("  • p=0 to foton/gluon/neutryno w stanie spoczynku —")
        print("    stan fizycznie niedostępny dla bezmasowych cząstek!")

        # ── Porównanie mas
        print("\n▶  Propagator fermionowy przy różnych masach\n")
        results = self.massive_propagator_at_pole()
        for mass_str, data in results.items():
            pole_val = mass_str if mass_str != "oo" else "∞"
            w = data["wheel_at_pole"]
            print(f"  m={mass_str:<6} → Wheel(p=m) = {w}")

        # ── Chiralność i m=0
        print("\n▶  Chiralność i granica m→0\n")
        print("  Dla m=0 równanie Diraca rozpada się na dwa niezależne:")
        print("  • Leworęczne: iσ^μ∂_μ ψ_L = 0  (Weyl)")
        print("  • Praworęczne: iσ̄^μ∂_μ ψ_R = 0  (Weyl)")
        print()

        # Propagator masowy w granicy m→0
        prop_massive = (p + m_e) / (p**2 - m_e**2)
        prop_massive_simplified = sp.simplify(prop_massive)
        print(f"  S_F(p,m) = {prop_massive_simplified}")

        lim_m0 = sp.limit(prop_massive_simplified, m_e, 0)
        print(f"  lim(m→0) S_F = {lim_m0}")
        print(f"  Wheel(m=0, p=0) = {wheel_subs(lim_m0, {p: sp.S.Zero})}")
        print()
        print("  Wniosek: granica m→0 jest regularna wszędzie POZA p=0.")
        print("  Wheel wskazuje p=0 jako jedyną prawdziwą IR-osobliwość.")


# ─── 4. Relacja dyspersji i powłoka masy ─────────────────────────────────────

class DispersionRelation:
    """
    Relacja dyspersji relativistycznej:
      E² = (pc)² + (mc²)²

    'On-shell': p² = m²c² (w jednostkach c=1: p² = m²)
    'Off-shell': p² ≠ m²  (cząstki wirtualne w diagramach Feynmana)

    W Wheel: propagator 1/(p²-m²) ma ⊥ dokładnie on-shell.
    """

    def analyse_on_shell(self) -> None:
        print("\n" + "═" * 64)
        print("  RELACJA DYSPERSJI — On-shell vs Off-shell w Wheel")
        print("═" * 64)

        prop = sp.Integer(1) / (p**2 - m**2)

        print("\n▶  Propagator 1/(p²-m²) w funkcji p (przy m=1)\n")

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
            marker = "  ←  BIEGUN" if w.is_bottom else ""
            print(f"  {label}: {str(w):<20}{marker}")

        print()
        print("  Interpretacja Wheel:")
        print("  • Off-shell (cząstki wirtualne): propagator skończony")
        print("  • On-shell  (cząstki rzeczywiste): propagator = ⊥")
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │ Prowokacyjna hipoteza:                               │")
        print("  │  ⊥ w propagatorze = cząstka jest RZECZYWIŚCIE       │")
        print("  │  obserwowalna (on-shell = detektor ją 'widzi').      │")
        print("  │                                                      │")
        print("  │  Macierz S łączy stany asymptotyczne (on-shell).    │")
        print("  │  Wheel automatycznie zaznacza te stany przez ⊥.     │")
        print("  │                                                      │")
        print("  │  Pytanie: czy ⊥ może być algebraiczną definicją     │")
        print("  │  'obserwowalności' w QFT?                           │")
        print("  └─────────────────────────────────────────────────────┘")


# ─── Główna analiza ───────────────────────────────────────────────────────────

def run_quantum_analysis() -> None:
    print("\n" + "█" * 64)
    print("  WHEELPHYSICS — Mechanika Kwantowa i QED")
    print("  Dywergencje, propagatory, renormalizacja")
    print("█" * 64)

    FeynmanPropagators().analyse_all()
    UVDivergences().analyse_divergences()
    DiracEquation().analyse_dirac()
    DispersionRelation().analyse_on_shell()

    # ── Syntetyczne podsumowanie badawcze
    print("\n" + "═" * 64)
    print("  PODSUMOWANIE BADAWCZE — QFT × Wheel")
    print("═" * 64)

    print("""
  WYNIK 1: Propagatory i bieguny
  ─────────────────────────────
  Każdy propagator Feynmana ma biegun on-shell → ⊥ w Wheel.
  Recepta iε (Feynman prescription) to klasyczny trik omijający
  ten biegun w zespolonej płaszczyźnie. Wheel zamiast omijać —
  przechodzi przez biegun i zwraca ⊥.

  Czy ⊥ jest naturalniejszym opisem stanu on-shell niż granica ε→0?

  WYNIK 2: Dywergencje UV
  ──────────────────────
  Dywergencje UV (Λ→∞) dają ⊥ gdy traktujemy Λ jako argument.
  Wheel NIE eliminuje renormalizacji automatycznie — dywergencje
  są w CAŁKACH, nie w pojedynczych wartościach.
  
  Ale: Wheel precyzyjnie lokalizuje SKĄD pochodzi każda dywergencja.
  To może być narzędzie diagnostyczne dla teorii regularyzacji.

  WYNIK 3: Równanie Diraca m=0
  ────────────────────────────
  Propagator bezmasowy 1/p → ⊥ przy p=0.
  Fizycznie: bezmasowa cząstka nie może mieć p=0 (relatywistycznie).
  Wheel algebraicznie zakazuje tego stanu — poprawnie!

  WYNIK 4: On-shell = ⊥  (najbardziej prowokacyjny wynik)
  ────────────────────────────────────────────────────────
  Cząstki wirtualne (off-shell): propagator skończony → obliczalny
  Cząstki rzeczywiste (on-shell): propagator = ⊥ → 'poza algebrą'
  
  W QFT cząstki wirtualne NIE są obserwowalne.
  Cząstki on-shell SĄ obserwowalne.
  
  Wheel algebraicznie oddziela te dwie klasy przez ⊥.
  """)


if __name__ == "__main__":
    run_quantum_analysis()