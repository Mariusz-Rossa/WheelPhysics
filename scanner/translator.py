# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
translator.py — tłumacz wyrażeń klasycznych → Wheel Algebra

Przyjmuje wyrażenie SymPy i przepisuje je do postaci Wheel,
identyfikując gdzie potrzeba inwersji wheel i gdzie pojawi się ⊥.

Pipeline:
  1. Parsuj wyrażenie
  2. Znajdź wszystkie podwyrażenia z dzieleniem
  3. Zastąp x/y → x * /y  (notacja wheel)
  4. Zaznacz miejsca gdzie /0 → ⊥
  5. Zwróć raport z tłumaczeniem
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Optional
import sympy as sp

from core.wheel_number import WheelNumber, BOTTOM, W
from core.wheel_algebra import WheelAlgebra
from core.sympy_extension import wheel_subs, is_singular_at, WheelFunction


_wa = WheelAlgebra()


# ─── Wynik tłumaczenia ────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    """Wynik przepisania wyrażenia klasycznego do Wheel."""
    original:          sp.Basic
    wheel_repr:        str              # notacja wheel (tekstowa)
    division_sites:    list[dict]       # każde miejsce dzielenia
    singular_points:   list[dict]       # punkty gdzie wynik = ⊥
    is_wheel_trivial:  bool             # True jeśli nie ma dzielenia (identyczne)
    notes:             list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "─" * 62,
            f"ORYGINAŁ    : {self.original}",
            f"WHEEL       : {self.wheel_repr}",
            f"Trywialny   : {'TAK (brak dzielenia)' if self.is_wheel_trivial else 'NIE — dzielenie wykryte'}",
        ]

        if self.division_sites:
            lines.append(f"\nMIEJSCA DZIELENIA ({len(self.division_sites)}):")
            for i, site in enumerate(self.division_sites, 1):
                lines.append(f"  [{i}] {site['numerator']} / {site['denominator']}")
                lines.append(f"       Wheel: {site['numerator']} * /({site['denominator']})")
                if site.get("denominator_zeros"):
                    zeros = ", ".join(str(z) for z in site["denominator_zeros"])
                    lines.append(f"       Zeruje się gdy: {zeros}")

        if self.singular_points:
            lines.append(f"\nPUNKTY OSOBLIWE ({len(self.singular_points)}):")
            for sp_pt in self.singular_points:
                lines.append(
                    f"  {sp_pt['variable']} = {sp_pt['value']}"
                    f"  →  klasycznie: {sp_pt['classical']}"
                    f"  |  Wheel: ⊥"
                )

        if self.notes:
            lines.append("\nUWAGI:")
            for note in self.notes:
                lines.append(f"  • {note}")

        lines.append("─" * 62)
        return "\n".join(lines)


# ─── Translator ───────────────────────────────────────────────────────────────

class Translator:
    """
    Tłumaczy wyrażenia klasyczne → Wheel Algebra.

    Użycie:
        t = Translator()
        result = t.translate(expr, variables=[r], check_values=[0, r_s])
        print(result.report())
    """

    def translate(
        self,
        expr: sp.Basic,
        variables: Optional[list[sp.Symbol]] = None,
        check_values: Optional[list] = None,
        name: str = "",
    ) -> TranslationResult:
        """
        Główna metoda tłumaczenia.

        Args:
            expr:         wyrażenie SymPy
            variables:    zmienne do sprawdzenia
            check_values: wartości do przetestowania (czy dają ⊥)
            name:         opcjonalna nazwa wyrażenia
        """
        expr = sp.sympify(expr)

        if variables is None:
            variables = sorted(expr.free_symbols, key=str)

        if check_values is None:
            check_values = [sp.S.Zero]

        # 1. Znajdź miejsca dzielenia
        division_sites = self._find_division_sites(expr, variables)

        # 2. Wygeneruj notację wheel
        wheel_repr = self._to_wheel_notation(expr)

        # 3. Sprawdź punkty osobliwe
        singular_points = []
        for var in variables:
            for val in check_values:
                if is_singular_at(expr, var, val):
                    classical = self._classical_limit(expr, var, val)
                    singular_points.append({
                        "variable":  var,
                        "value":     val,
                        "classical": classical,
                    })

        # 4. Czy tłumaczenie jest trywialne
        is_trivial = len(division_sites) == 0

        # 5. Uwagi
        notes = []
        if is_trivial:
            notes.append("Brak dzielenia — Wheel daje identyczny wynik jak klasyczna algebra")
        if len(singular_points) > 0:
            notes.append(
                f"Znaleziono {len(singular_points)} punkt(ów) osobliwych — "
                f"Wheel przypisuje im wartość ⊥"
            )

        return TranslationResult(
            original=expr,
            wheel_repr=wheel_repr,
            division_sites=division_sites,
            singular_points=singular_points,
            is_wheel_trivial=is_trivial,
            notes=notes,
        )

    def _find_division_sites(
        self,
        expr: sp.Basic,
        variables: list[sp.Symbol],
    ) -> list[dict]:
        """Wykrywa wszystkie miejsce dzielenia w wyrażeniu."""
        sites = []
        seen_denoms = set()

        def walk(e):
            numer, denom = sp.fraction(e)
            if denom != sp.S.One and str(denom) not in seen_denoms:
                seen_denoms.add(str(denom))

                # Znajdź gdzie mianownik = 0
                zeros = []
                for var in variables:
                    try:
                        sols = sp.solve(denom, var)
                        zeros.extend([(var, s) for s in sols])
                    except Exception:
                        pass

                sites.append({
                    "subexpr":         e,
                    "numerator":       numer,
                    "denominator":     denom,
                    "denominator_zeros": [f"{v}={z}" for v, z in zeros],
                })

            for arg in e.args:
                walk(arg)

        walk(expr)
        return sites

    def _to_wheel_notation(self, expr: sp.Basic) -> str:
        """
        Przepisuje wyrażenie do notacji wheel.
        x/y → x·/(y), /0 → ⊥
        """
        def rewrite(e) -> str:
            numer, denom = sp.fraction(e)

            if denom != sp.S.One:
                n_str = str(numer)
                d_str = str(denom)
                if denom == sp.S.Zero:
                    return f"{n_str}·/(0) = ⊥"
                return f"{n_str}·/({d_str})"

            # Rekurencja dla złożonych wyrażeń
            if e.args:
                return str(e)
            return str(e)

        return rewrite(expr)

    def _classical_limit(self, expr, var, val) -> str:
        try:
            lim = sp.limit(expr, var, val)
            return str(lim)
        except Exception:
            return "∞ lub nieoznaczone"

    # ── Batch translate ───────────────────────────────────────────────────────

    def translate_many(
        self,
        equations: list[dict],
    ) -> list[TranslationResult]:
        """
        Tłumaczy listę równań.

        Args:
            equations: lista słowników z kluczami:
                       "expr", "variables", "check_values", "name"
        """
        results = []
        for eq in equations:
            result = self.translate(
                eq["expr"],
                variables=eq.get("variables"),
                check_values=eq.get("check_values", [sp.S.Zero]),
                name=eq.get("name", ""),
            )
            results.append(result)
        return results


# ─── Predefiniowane tłumaczenia znanych równań ────────────────────────────────

def translate_schwarzschild() -> list[TranslationResult]:
    """Tłumaczy składowe metryki Schwarzschilda."""
    r, r_s = sp.symbols("r r_s", positive=True)
    t = Translator()

    return t.translate_many([
        {
            "name": "g_tt (Schwarzschild)",
            "expr": -(1 - r_s/r),
            "variables": [r],
            "check_values": [sp.S.Zero, r_s],
        },
        {
            "name": "g_rr (Schwarzschild)",
            "expr": 1/(1 - r_s/r),
            "variables": [r],
            "check_values": [sp.S.Zero, r_s],
        },
    ])


def translate_friedmann() -> TranslationResult:
    """Tłumaczy człon Friedmanna z osobliwością przy a=0."""
    a, k, c = sp.symbols("a k c")
    t = Translator()
    return t.translate(
        k * c**2 / a**2,
        variables=[a],
        check_values=[sp.S.Zero],
        name="Friedmann k·c²/a²",
    )


def translate_feynman_propagator() -> TranslationResult:
    """Tłumaczy propagator Feynmana."""
    p, m = sp.symbols("p m", positive=True)
    t = Translator()
    return t.translate(
        1 / (p**2 - m**2),
        variables=[p],
        check_values=[m, -m, sp.S.Zero],
        name="Propagator Feynmana",
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 62)
    print("  Translator — klasyczna algebra → Wheel")
    print("═" * 62)

    print("\n▶  Metryka Schwarzschilda")
    for result in translate_schwarzschild():
        print(result.report())

    print("\n▶  Równanie Friedmanna")
    print(translate_friedmann().report())

    print("\n▶  Propagator Feynmana")
    print(translate_feynman_propagator().report())

    # Wyrażenie bez dzielenia — test trywialności
    print("\n▶  Wyrażenie bez dzielenia (powinno być trywialne)")
    x, y = sp.symbols("x y")
    t = Translator()
    print(t.translate(x**2 + 2*x*y + y**2, variables=[x, y]).report())