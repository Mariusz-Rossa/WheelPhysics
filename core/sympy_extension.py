# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
sympy_extension.py — most między SymPy a Wheel Algebra

Pozwala używać standardowych wyrażeń SymPy i automatycznie
wykrywać gdzie trzeba użyć Wheel zamiast klasycznej arytmetyki.

Główne funkcje:
  - expr_to_wheel()     : konwertuje wyrażenie SymPy → WheelNumber
  - wheel_subs()        : bezpieczne podstawienie (nigdy nie rzuca wyjątku)
  - is_singular_at()    : szybkie sprawdzenie czy wyrażenie ma osobliwość
  - wheel_series()      : rozwinięcie w szereg w okolicy ⊥
  - WheelFunction       : wrapper dla funkcji f(x) z obsługą ⊥
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union
import sympy as sp

from core.wheel_number import WheelNumber, BOTTOM, W, _coerce
from core.wheel_algebra import WheelAlgebra


_wa = WheelAlgebra()


# ─── Konwersja wyrażeń SymPy ──────────────────────────────────────────────────

def expr_to_wheel(expr: sp.Basic) -> WheelNumber:
    """
    Konwertuje wyrażenie SymPy do WheelNumber.

    Jeśli wyrażenie zawiera nieskończoność, NaN lub zoo → ⊥.
    Wyrażenia symboliczne są zachowane jako-jest (ewaluacja leniwa).
    """
    expr = sp.sympify(expr)

    if expr in (sp.oo, sp.zoo, sp.nan, -sp.oo):
        return W(BOTTOM)

    # Sprawdź czy wyrażenie jest numerycznie nieskończone
    try:
        evaled = expr.evalf()
        if evaled in (sp.oo, sp.zoo, sp.nan, -sp.oo):
            return W(BOTTOM)
        # Sprawdź float overflow
        if evaled.is_number:
            f = float(evaled)
            if not (f == f) or abs(f) == float('inf'):  # nan or inf
                return W(BOTTOM)
    except (TypeError, ValueError, OverflowError):
        pass  # Wyrażenie symboliczne — OK
    except Exception:
        pass

    return W(expr)


def wheel_subs(
    expr: sp.Basic,
    substitutions: dict,
    trace: bool = False,
) -> WheelNumber:
    """
    Bezpieczne podstawienie wartości do wyrażenia SymPy.
    Nigdy nie rzuca wyjątku — zamiast tego zwraca ⊥.

    Args:
        expr:          wyrażenie SymPy
        substitutions: {symbol: wartość}
        trace:         drukuj co się dzieje

    Przykład:
        r, r_s = sp.symbols("r r_s")
        g_rr = 1 / (1 - r_s/r)
        wheel_subs(g_rr, {r: r_s})  →  W(⊥)
    """
    w = expr_to_wheel(expr)
    return _wa.evaluate_at(w, substitutions, trace=trace)


def is_singular_at(
    expr: sp.Basic,
    var: sp.Symbol,
    value,
) -> bool:
    """
    Szybkie sprawdzenie czy wyrażenie ma osobliwość przy var=value.

    Returns:
        True  — jest osobliwość (Wheel da ⊥)
        False — brak osobliwości
    """
    result = wheel_subs(expr, {var: value})
    return result.is_bottom


def singularity_map(
    expr: sp.Basic,
    var: sp.Symbol,
    values: list,
) -> dict:
    """
    Mapuje listę wartości na wyniki Wheel.
    Pokazuje gdzie wyrażenie jest regularne, a gdzie daje ⊥.

    Returns:
        {value: WheelNumber}
    """
    return {v: wheel_subs(expr, {var: v}) for v in values}


# ─── Rozwinięcie w okolicy osobliwości ───────────────────────────────────────

def wheel_series_around(
    expr: sp.Basic,
    var: sp.Symbol,
    point,
    n_terms: int = 4,
) -> dict:
    """
    Analizuje zachowanie wyrażenia w okolicy punktu osobliwego.

    Dla każdego epsilon w {+, -} sprawdza:
      - wartość przy point ± epsilon dla małych epsilon
      - czy istnieje granica (i jaka)
      - wynik Wheel przy dokładnym point

    Returns:
        dict z wynikami analizy
    """
    point = sp.sympify(point)
    epsilon = sp.Symbol("epsilon", positive=True)

    result = {
        "expression":    expr,
        "variable":      var,
        "point":         point,
        "wheel_at_point": wheel_subs(expr, {var: point}),
        "limit_from_right": None,
        "limit_from_left":  None,
        "series_right":     None,
        "series_left":      None,
    }

    # Granice
    try:
        result["limit_from_right"] = sp.limit(expr, var, point, "+")
    except Exception:
        result["limit_from_right"] = "nie istnieje"

    try:
        result["limit_from_left"] = sp.limit(expr, var, point, "-")
    except Exception:
        result["limit_from_left"] = "nie istnieje"

    # Szereg Laurenta (jeśli możliwy)
    try:
        series = sp.series(expr, var, point, n=n_terms)
        result["laurent_series"] = series
    except Exception:
        result["laurent_series"] = "nie daje się rozwinąć"

    return result


def print_singularity_analysis(analysis: dict) -> None:
    """Ładny wydruk analizy z wheel_series_around."""
    print(f"  Wyrażenie    : {analysis['expression']}")
    print(f"  Zmienna      : {analysis['variable']} → {analysis['point']}")
    print(f"  Wheel(punkt) : {analysis['wheel_at_point']}")
    print(f"  Granica (+)  : {analysis['limit_from_right']}")
    print(f"  Granica (-)  : {analysis['limit_from_left']}")
    if "laurent_series" in analysis:
        print(f"  Laurent      : {analysis['laurent_series']}")


# ─── WheelFunction — wrapper dla funkcji symbolicznych ───────────────────────

class WheelFunction:
    """
    Wrapper dla funkcji symbolicznej f(x) z obsługą ⊥.

    Pozwala ewaluować funkcje SymPy w sposób bezpieczny —
    zamiast wyjątku lub ∞ zwraca ⊥.

    Przykład:
        g_rr = WheelFunction(1/(1 - r_s/r), r, name="g_rr Schwarzschild")
        g_rr(r_s)    →  ⊥
        g_rr(2*r_s)  →  -1
    """

    def __init__(
        self,
        expr: sp.Basic,
        *variables: sp.Symbol,
        name: str = "f",
    ):
        self.expr = sp.sympify(expr)
        self.variables = variables
        self.name = name

    def __call__(self, *values) -> WheelNumber:
        if len(values) != len(self.variables):
            raise ValueError(
                f"Oczekiwano {len(self.variables)} argumentów, "
                f"dostałem {len(values)}"
            )
        subs = dict(zip(self.variables, values))
        return wheel_subs(self.expr, subs)

    def scan_range(self, var: sp.Symbol, values: list) -> list[tuple]:
        """
        Ewaluuje funkcję dla listy wartości.
        Returns: [(value, WheelNumber), ...]
        """
        results = []
        other_vars = [v for v in self.variables if v != var]
        for val in values:
            subs = {var: val}
            result = wheel_subs(self.expr, subs)
            results.append((val, result))
        return results

    def find_singularities_in_range(
        self,
        var: sp.Symbol,
        values: list,
    ) -> list:
        """Zwraca tylko te wartości gdzie wynik = ⊥."""
        return [
            (v, r) for v, r in self.scan_range(var, values)
            if r.is_bottom
        ]

    def __repr__(self) -> str:
        return f"WheelFunction({self.name}: {self.expr})"


# ─── Testy i demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  sympy_extension — integracja SymPy + Wheel")
    print("═" * 60)

    r, r_s, m, p = sp.symbols("r r_s m p", positive=True)

    # ── Test 1: expr_to_wheel
    print("\n▶  expr_to_wheel")
    cases = [
        (sp.Integer(5),           "5"),
        (sp.oo,                   "∞ → ⊥"),
        (sp.zoo,                  "zoo → ⊥"),
        (sp.nan,                  "nan → ⊥"),
        (1/(r - r_s),             "1/(r-r_s) symboliczne"),
    ]
    for expr, desc in cases:
        result = expr_to_wheel(expr)
        print(f"  {desc:<30} → {result}")

    # ── Test 2: wheel_subs na metryce Schwarzschilda
    print("\n▶  wheel_subs — g_rr Schwarzschilda przy różnych r")
    g_rr_expr = 1 / (1 - r_s / r)
    test_values = {
        "r = 2·r_s":  {r: 2*r_s},
        "r = r_s":    {r: r_s},
        "r = r_s/2":  {r: r_s/2},
    }
    for desc, subs in test_values.items():
        result = wheel_subs(g_rr_expr, subs)
        print(f"  {desc:<15} → {result}")

    # ── Test 3: WheelFunction
    print("\n▶  WheelFunction — g_rr Schwarzschilda")
    g_rr_fn = WheelFunction(1/(1 - r_s/r), r, name="g_rr")
    print(f"  {g_rr_fn}")
    for val, label in [(2*r_s, "2·r_s"), (r_s, "r_s"), (r_s/2, "r_s/2")]:
        print(f"  g_rr({label}) = {g_rr_fn(val)}")

    # ── Test 4: Analiza osobliwości
    print("\n▶  Analiza osobliwości: 1/(p²-m²) przy p=m")
    propagator = 1 / (p**2 - m**2)
    analysis = wheel_series_around(propagator, p, m)
    print_singularity_analysis(analysis)

    # ── Test 5: singularity_map
    print("\n▶  Mapa osobliwości: 1/r dla r ∈ {-1, 0, 1, 2}")
    x = sp.Symbol("x")
    smap = singularity_map(1/x, x, [-1, 0, 1, 2])
    for val, result in smap.items():
        marker = " ← ⊥ (osobliwość)" if result.is_bottom else ""
        print(f"  x={val:>2} → {result}{marker}")

    print("\n" + "═" * 60)