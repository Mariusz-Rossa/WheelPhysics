# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
translator.py — classical expression translator → Wheel Algebra

Takes a SymPy expression and rewrites it into the Wheel form,
identifying where wheel inversion is needed and where ⊥ will appear.

Pipeline:
  1. Parse expression
  2. Find all subexpressions containing division
  3. Replace x/y → x * /y  (wheel notation)
  4. Mark positions where /0 → ⊥
  5. Return translation report
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


# ─── Translation result ───────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    """The result of rewriting a classical expression into Wheel."""
    original:          sp.Basic
    wheel_repr:        str              # wheel notation (textual)
    division_sites:    list[dict]       # each division site
    singular_points:   list[dict]       # points where result = ⊥
    is_wheel_trivial:  bool             # True if there is no division (identical)
    notes:             list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "─" * 62,
            f"ORIGINAL    : {self.original}",
            f"WHEEL       : {self.wheel_repr}",
            f"Trivial     : {'YES (no division)' if self.is_wheel_trivial else 'NO — division detected'}",
        ]

        if self.division_sites:
            lines.append(f"\nDIVISION SITES ({len(self.division_sites)}):")
            for i, site in enumerate(self.division_sites, 1):
                lines.append(f"  [{i}] {site['numerator']} / {site['denominator']}")
                lines.append(f"       Wheel: {site['numerator']} * /({site['denominator']})")
                if site.get("denominator_zeros"):
                    zeros = ", ".join(str(z) for z in site["denominator_zeros"])
                    lines.append(f"       Zeroes out when: {zeros}")

        if self.singular_points:
            lines.append(f"\nSINGULAR POINTS ({len(self.singular_points)}):")
            for sp_pt in self.singular_points:
                lines.append(
                    f"  {sp_pt['variable']} = {sp_pt['value']}"
                    f"  →  classical: {sp_pt['classical']}"
                    f"  |  Wheel: ⊥"
                )

        if self.notes:
            lines.append("\nNOTES:")
            for note in self.notes:
                lines.append(f"  • {note}")

        lines.append("─" * 62)
        return "\n".join(lines)


# ─── Translator ───────────────────────────────────────────────────────────────

class Translator:
    """
    Translates classical expressions → Wheel Algebra.

    Usage:
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
        Main translation method.

        Args:
            expr:         SymPy expression
            variables:    variables to verify
            check_values: values to test (whether they yield ⊥)
            name:         optional expression name
        """
        expr = sp.sympify(expr)

        if variables is None:
            variables = sorted(expr.free_symbols, key=str)

        if check_values is None:
            check_values = [sp.S.Zero]

        # 1. Find division sites
        division_sites = self._find_division_sites(expr, variables)

        # 2. Generate wheel notation
        wheel_repr = self._to_wheel_notation(expr)

        # 3. Check singular points
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

        # 4. Is the translation trivial
        is_trivial = len(division_sites) == 0

        # 5. Notes
        notes = []
        if is_trivial:
            notes.append("No division — Wheel yields an identical result to classical algebra")
        if len(singular_points) > 0:
            notes.append(
                f"Found {len(singular_points)} singular point(s) — "
                f"Wheel assigns them the value ⊥"
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
        """Detects all division sites in an expression."""
        fields = []
        seen_denoms = set()

        def walk(e):
            numer, denom = sp.fraction(e)
            if denom != sp.S.One and str(denom) not in seen_denoms:
                seen_denoms.add(str(denom))

                # Find where denominator = 0
                zeros = []
                for var in variables:
                    try:
                        sols = sp.solve(denom, var)
                        zeros.extend([(var, s) for s in sols])
                    except Exception:
                        pass

                fields.append({
                    "subexpr":         e,
                    "numerator":       numer,
                    "denominator":     denom,
                    "denominator_zeros": [f"{v}={z}" for v, z in zeros],
                })

            for arg in e.args:
                walk(arg)

        walk(expr)
        return fields

    def _to_wheel_notation(self, expr: sp.Basic) -> str:
        """
        Rewrites an expression to wheel notation.
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

            # Recursion for complex expressions
            if e.args:
                return str(e)
            return str(e)

        return rewrite(expr)

    def _classical_limit(self, expr, var, val) -> str:
        try:
            lim = sp.limit(expr, var, val)
            return str(lim)
        except Exception:
            return "∞ or indeterminate"

    # ── Batch translate ───────────────────────────────────────────────────────

    def translate_many(
        self,
        equations: list[dict],
    ) -> list[TranslationResult]:
        """
        Translates a list of equations.

        Args:
            equations: list of dictionaries with keys:
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


# ─── Predefined translations of known equations ──────────────────────────────

def translate_schwarzschild() -> list[TranslationResult]:
    """Translates Schwarzschild metric components."""
    r, r_s = sp.symbols("r r_s", positive=True)
    t = Translator()

    return t.translate_many([
        {
            "name": "Schwarzschild g_tt",
            "expr": -(1 - r_s/r),
            "variables": [r],
            "check_values": [sp.S.Zero, r_s],
        },
        {
            "name": "Schwarzschild g_rr",
            "expr": 1/(1 - r_s/r),
            "variables": [r],
            "check_values": [sp.S.Zero, r_s],
        },
    ])


def translate_friedmann() -> TranslationResult:
    """Translates Friedmann term with singularity at a=0."""
    a, k, c = sp.symbols("a k c")
    t = Translator()
    return t.translate(
        k * c**2 / a**2,
        variables=[a],
        check_values=[sp.S.Zero],
        name="Friedmann k·c²/a²",
    )


def translate_feynman_propagator() -> TranslationResult:
    """Translates Feynman propagator."""
    p, m = sp.symbols("p m", positive=True)
    t = Translator()
    return t.translate(
        1 / (p**2 - m**2),
        variables=[p],
        check_values=[m, -m, sp.S.Zero],
        name="Feynman propagator",
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 62)
    print("  Translator — classical algebra → Wheel")
    print("═" * 62)

    print("\n▶  Schwarzschild Metric")
    for result in translate_schwarzschild():
        print(result.report())

    print("\n▶  Friedmann Equation")
    print(translate_friedmann().report())

    print("\n▶  Feynman Propagator")
    print(translate_feynman_propagator().report())

    # Expression without division — triviality test
    print("\n▶  Expression without division (should be trivial)")
    x, y = sp.symbols("x y")
    t = Translator()
    print(t.translate(x**2 + 2*x*y + y**2, variables=[x, y]).report())