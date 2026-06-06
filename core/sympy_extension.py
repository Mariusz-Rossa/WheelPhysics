# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
sympy_extension.py — bridge between SymPy and Wheel Algebra

Allows you to use standard SymPy expressions and automatically
detect where Wheel needs to be used instead of classical arithmetic.

Main functions:
  - expr_to_wheel()     : converts a SymPy expression → WheelNumber
  - wheel_subs()        : safe substitution (never throws an exception)
  - is_singular_at()    : quick check if an expression has a singularity
  - wheel_series()      : series expansion around ⊥
  - WheelFunction       : wrapper for function f(x) with ⊥ support
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union
import sympy as sp

from core.wheel_number import WheelNumber, BOTTOM, W, _coerce
from core.wheel_algebra import WheelAlgebra


_wa = WheelAlgebra()


# ─── SymPy Expression Conversion ──────────────────────────────────────────────

def expr_to_wheel(expr: sp.Basic) -> WheelNumber:
    """
    Converts a SymPy expression to a WheelNumber.

    If the expression contains infinity, NaN, or zoo → ⊥.
    Symbolic expressions are kept as-is (lazy evaluation).
    """
    expr = sp.sympify(expr)

    if expr in (sp.oo, sp.zoo, sp.nan, -sp.oo):
        return W(BOTTOM)

    # Check if the expression is numerically infinite
    try:
        evaled = expr.evalf()
        if evaled in (sp.oo, sp.zoo, sp.nan, -sp.oo):
            return W(BOTTOM)
        # Check float overflow
        if evaled.is_number:
            f = float(evaled)
            if not (f == f) or abs(f) == float('inf'):  # nan or inf
                return W(BOTTOM)
    except (TypeError, ValueError, OverflowError):
        pass  # Symbolic expression — OK
    except Exception:
        pass

    return W(expr)


def wheel_subs(
    expr: sp.Basic,
    substitutions: dict,
    trace: bool = False,
) -> WheelNumber:
    """
    Safe substitution of values into a SymPy expression.
    Never throws an exception — returns ⊥ instead.

    Args:
        expr:          SymPy expression
        substitutions: {symbol: value}
        trace:         print what is happening

    Example:
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
    Quick check if the expression has a singularity at var=value.

    Returns:
        True  — singularity present (Wheel gives ⊥)
        False — no singularity
    """
    result = wheel_subs(expr, {var: value})
    return result.is_bottom


def singularity_map(
    expr: sp.Basic,
    var: sp.Symbol,
    values: list,
) -> dict:
    """
    Maps a list of values to Wheel results.
    Shows where the expression is regular and where it yields ⊥.

    Returns:
        {value: WheelNumber}
    """
    return {v: wheel_subs(expr, {var: v}) for v in values}


# ─── Expansion around singularity ────────────────────────────────────────────

def wheel_series_around(
    expr: sp.Basic,
    var: sp.Symbol,
    point,
    n_terms: int = 4,
) -> dict:
    """
    Analyzes the behavior of an expression around a singular point.

    For each epsilon in {+, -} it checks:
      - value at point ± epsilon for small epsilon
      - whether a limit exists (and what it is)
      - Wheel result at exact point

    Returns:
        dict with analysis results
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

    # Limits
    try:
        result["limit_from_right"] = sp.limit(expr, var, point, "+")
    except Exception:
        result["limit_from_right"] = "does not exist"

    try:
        result["limit_from_left"] = sp.limit(expr, var, point, "-")
    except Exception:
        result["limit_from_left"] = "does not exist"

    # Laurent series (if possible)
    try:
        series = sp.series(expr, var, point, n=n_terms)
        result["laurent_series"] = series
    except Exception:
        result["laurent_series"] = "cannot be expanded"

    return result


def print_singularity_analysis(analysis: dict) -> None:
    """Pretty print for analysis from wheel_series_around."""
    print(f"  Expression   : {analysis['expression']}")
    print(f"  Variable     : {analysis['variable']} → {analysis['point']}")
    print(f"  Wheel(point) : {analysis['wheel_at_point']}")
    print(f"  Limit (+)    : {analysis['limit_from_right']}")
    print(f"  Limit (-)    : {analysis['limit_from_left']}")
    if "laurent_series" in analysis:
        print(f"  Laurent      : {analysis['laurent_series']}")


# ─── WheelFunction — wrapper for symbolic functions ──────────────────────────

class WheelFunction:
    """
    Wrapper for a symbolic function f(x) with ⊥ support.

    Allows SymPy functions to be evaluated safely —
    returns ⊥ instead of an exception or ∞.

    Example:
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
                f"Expected {len(self.variables)} arguments, "
                f"got {len(values)}"
            )
        subs = dict(zip(self.variables, values))
        return wheel_subs(self.expr, subs)

    def scan_range(self, var: sp.Symbol, values: list) -> list[tuple]:
        """
        Evaluates the function for a list of values.
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
        """Returns only those values where result = ⊥."""
        return [
            (v, r) for v, r in self.scan_range(var, values)
            if r.is_bottom
        ]

    def __repr__(self) -> str:
        return f"WheelFunction({self.name}: {self.expr})"


# ─── Tests and demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  sympy_extension — SymPy + Wheel integration")
    print("═" * 60)

    r, r_s, m, p = sp.symbols("r r_s m p", positive=True)

    # ── Test 1: expr_to_wheel
    print("\n▶  expr_to_wheel")
    cases = [
        (sp.Integer(5),           "5"),
        (sp.oo,                   "∞ → ⊥"),
        (sp.zoo,                  "zoo → ⊥"),
        (sp.nan,                  "nan → ⊥"),
        (1/(r - r_s),             "1/(r-r_s) symbolic"),
    ]
    for expr, desc in cases:
        result = expr_to_wheel(expr)
        print(f"  {desc:<30} → {result}")

    # ── Test 2: wheel_subs on Schwarzschild metric
    print("\n▶  wheel_subs — Schwarzschild g_rr at different r")
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
    print("\n▶  WheelFunction — Schwarzschild g_rr")
    g_rr_fn = WheelFunction(1/(1 - r_s/r), r, name="g_rr")
    print(f"  {g_rr_fn}")
    for val, label in [(2*r_s, "2·r_s"), (r_s, "r_s"), (r_s/2, "r_s/2")]:
        print(f"  g_rr({label}) = {g_rr_fn(val)}")

    # ── Test 4: Singularity analysis
    print("\n▶  Singularity analysis: 1/(p²-m²) at p=m")
    propagator = 1 / (p**2 - m**2)
    analysis = wheel_series_around(propagator, p, m)
    print_singularity_analysis(analysis)

    # ── Test 5: singularity_map
    print("\n▶  Singularity map: 1/r for r ∈ {-1, 0, 1, 2}")
    x = sp.Symbol("x")
    smap = singularity_map(1/x, x, [-1, 0, 1, 2])
    for val, result in smap.items():
        marker = " ← ⊥ (singularity)" if result.is_bottom else ""
        print(f"  x={val:>2} → {result}{marker}")

    print("\n" + "═" * 60)