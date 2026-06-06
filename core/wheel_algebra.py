# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
wheel_algebra.py — wheel algebra rules and expression rewriting

Implements formal expression rewriting rules in wheel algebra,
according to Carlström's (2004) axioms.

Key rules:
  - ⊥ detection and propagation
  - Wheel expression normalization
  - Reduction of indeterminate forms
  - Rules for wheel-style limits

Difference from classical algebra:
  Classical:  x*(y+z) = x*y + x*z          (always)
  Wheel:      x*(y+z) = x*y + x*z + 0*⊥   (when x, y, z ∉ {0, ⊥})
              — distributivity does NOT hold for 0 and ⊥
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import sympy as sp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.wheel_number import WheelNumber, WheelValue, BOTTOM, W, _coerce


# ─── Rewriting rules ──────────────────────────────────────────────────────────

@dataclass
class RewriteRule:
    """Single rewrite rule with description."""
    name: str
    description: str
    apply: Callable[[WheelNumber, Optional[WheelNumber]], Optional[WheelNumber]]



def _contains_infinity(expr) -> bool:
    """Checks if the expression contains zoo/oo/nan as a subexpression."""
    if expr in (sp.oo, sp.zoo, sp.nan, -sp.oo):
        return True
    try:
        return expr.has(sp.oo) or expr.has(sp.zoo) or expr.has(sp.nan)
    except Exception:
        return False



def _has_division_by_zero_at(expr: sp.Basic, substitutions: dict) -> bool:
    """
    Checks RECURSIVELY if division by zero appears anywhere
    in the expression after substitution.

    SymPy can algebraically simplify 1/(1-r_s/r) at r=0 to 0,
    losing the information about /r inside. This function walks
    the expression tree and checks every denominator before simplification.
    """
    def walk(e) -> bool:
        # Check the denominator of this subexpression
        n, d = sp.fraction(e)
        if d != sp.S.One:
            d_sub = sp.simplify(d.subs(substitutions))
            if d_sub == sp.S.Zero:
                return True
            if _contains_infinity(d_sub):
                return True

        # Recursion over arguments
        for arg in e.args:
            if walk(arg):
                return True
        return False

    try:
        return walk(sp.sympify(expr))
    except Exception:
        return False

class WheelAlgebra:
    """
    System of wheel algebra rules.

    Usage:
        wa = WheelAlgebra()
        result = wa.simplify(expr)
        wa.explain(expr)  # step-by-step
    """

    def __init__(self):
        self._rules = self._build_rules()

    # ── Rules construction ───────────────────────────────────────────────────

    def _build_rules(self) -> list[RewriteRule]:
        return [
            RewriteRule(
                name="bottom_absorb_add",
                description="⊥ + x = ⊥  (axiom 11)",
                apply=lambda a, b: W(BOTTOM) if (a.is_bottom or (b and b.is_bottom)) else None,
            ),
            RewriteRule(
                name="bottom_absorb_mul",
                description="⊥ * x = ⊥  (axiom 12)",
                apply=lambda a, b: W(BOTTOM) if (a.is_bottom or (b and b.is_bottom)) else None,
            ),
            RewriteRule(
                name="zero_inv",
                description="/0 = ⊥  (axiom 10: 0·/0 = ⊥)",
                apply=lambda a, b: W(BOTTOM) if a.is_zero else None,
            ),
            RewriteRule(
                name="double_inv",
                description="/(/x) = x  (axiom 8)",
                apply=lambda a, b: a,  # applied by wheel_inv().wheel_inv()
            ),
            RewriteRule(
                name="bottom_inv",
                description="/⊥ = ⊥  (⊥ is absorbing)",
                apply=lambda a, b: W(BOTTOM) if a.is_bottom else None,
            ),
            RewriteRule(
                name="zero_mul_bottom",
                description="0 * ⊥ = ⊥  (not 0!)",
                apply=lambda a, b: W(BOTTOM) if (
                    a.is_zero and b and b.is_bottom or
                    a.is_bottom and b and b.is_zero
                ) else None,
            ),
        ]

    # ── Algebraic operations with tracking ───────────────────────────────────

    def add(self, a: WheelValue, b: WheelValue, trace: bool = False) -> WheelNumber:
        a, b = _coerce(a), _coerce(b)
        if a.is_bottom or b.is_bottom:
            if trace:
                print(f"  ADD: ⊥ absorption → ⊥  [axiom 11]")
            return W(BOTTOM)
        result = a + b
        if trace:
            print(f"  ADD: {a} + {b} = {result}")
        return result

    def mul(self, a: WheelValue, b: WheelValue, trace: bool = False) -> WheelNumber:
        a, b = _coerce(a), _coerce(b)
        if a.is_bottom or b.is_bottom:
            if trace:
                print(f"  MUL: ⊥ absorption → ⊥  [axiom 12]")
            return W(BOTTOM)
        result = a * b
        if trace:
            print(f"  MUL: {a} * {b} = {result}")
        return result

    def inv(self, a: WheelValue, trace: bool = False) -> WheelNumber:
        """Multiplicative inversion /a."""
        a = _coerce(a)
        if a.is_bottom:
            if trace:
                print(f"  INV: /⊥ = ⊥  [⊥ absorption]")
            return W(BOTTOM)
        if a.is_zero:
            if trace:
                print(f"  INV: /0 = ⊥  [axiom 10]")
            return W(BOTTOM)
        result = a.wheel_inv()
        if trace:
            print(f"  INV: /{a} = {result}")
        return result

    def div(self, a: WheelValue, b: WheelValue, trace: bool = False) -> WheelNumber:
        """Division: a/b = a * /b."""
        a, b = _coerce(a), _coerce(b)
        b_inv = self.inv(b, trace=trace)
        return self.mul(a, b_inv, trace=trace)

    # ── Expression normalization ──────────────────────────────────────────────

    def normalize(self, expr: WheelNumber) -> WheelNumber:
        """
        Attempt to simplify a wheel expression.
        For symbolic expressions, uses SymPy.simplify.
        """
        if expr.is_bottom:
            return expr
        try:
            simplified = sp.simplify(expr.value)
            # Check if simplification introduced infinity
            if simplified in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                return W(BOTTOM)
            return W(simplified)
        except Exception:
            return expr

    # ── Evaluation of expression with substitution ────────────────────────────

    def evaluate_at(
        self,
        expr: WheelNumber,
        substitutions: dict,
        trace: bool = False,
    ) -> WheelNumber:
        """
        Substitutes values into a symbolic expression.
        Returns ⊥ if substitution leads to /0.

        Args:
            expr:          WheelNumber with SymPy symbols
            substitutions: {symbol: value}, e.g., {r: 0}
            trace:         whether to print steps
        """
        if expr.is_bottom:
            return expr

        if trace:
            print(f"\n  Evaluation: {expr.value}")
            print(f"  Substitution: {substitutions}")

        try:
            # KEY: recursively check denominators BEFORE substitution
            # (SymPy can simplify 1/(1-r_s/r) at r=0 to 0, losing info about /r)
            if _has_division_by_zero_at(expr.value, substitutions):
                if trace:
                    print(f"  → Found /0 in subexpression → ⊥")
                return W(BOTTOM)

            substituted = expr.value.subs(substitutions)

            # Check if result is infinite / undefined
            if substituted in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                if trace:
                    print(f"  → Classical result: {substituted} → mapping to ⊥")
                return W(BOTTOM)

            # Check if result CONTAINS zoo/oo as a factor (e.g., zoo*k)
            if _contains_infinity(substituted):
                if trace:
                    print(f"  → Contains infinity ({substituted}) → ⊥")
                return W(BOTTOM)

            # Check denominators after substitution
            numer, denom = sp.fraction(sp.simplify(substituted))
            if denom == sp.S.Zero or (hasattr(denom, 'is_zero') and denom.is_zero):
                if trace:
                    print(f"  → Denominator = 0 → ⊥")
                return W(BOTTOM)

            result = W(substituted)
            if trace:
                print(f"  → Result: {result}")
            return result

        except (ZeroDivisionError, sp.core.sympify.SympifyError):
            if trace:
                print(f"  → Division exception → ⊥")
            return W(BOTTOM)
        except Exception as e:
            if trace:
                print(f"  → Error: {e} → ⊥")
            return W(BOTTOM)

    # ── Distributivity — key difference vs classical algebra ─────────────────

    def distributivity_check(
        self,
        x: WheelValue,
        y: WheelValue,
        z: WheelValue,
        trace: bool = True,
    ) -> dict:
        """
        Checks if x*(y+z) = x*y + x*z in wheel algebra.

        In classical algebra, this is always true.
        In wheel: it does NOT hold when x=0 or the result contains ⊥.
        """
        x, y, z = _coerce(x), _coerce(y), _coerce(z)

        lhs = self.mul(x, self.add(y, z))         # x*(y+z)
        rhs = self.add(self.mul(x, y), self.mul(x, z))  # x*y + x*z

        holds = (lhs == rhs)

        if trace:
            print(f"\n  Distributivity: x*(y+z) = x*y + x*z ?")
            print(f"  x={x}, y={y}, z={z}")
            print(f"  LHS: {x}*({y}+{z}) = {x}*{self.add(y,z)} = {lhs}")
            print(f"  RHS: {x}*{y} + {x}*{z} = {self.mul(x,y)} + {self.mul(x,z)} = {rhs}")
            print(f"  Holds: {'✓ YES' if holds else '✗ NO — this distinguishes Wheel from classical algebra!'}")

        return {"lhs": lhs, "rhs": rhs, "holds": holds}

    # ── Wheel limit — wheel-style limit ──────────────────────────────────────

    def wheel_limit(
        self,
        expr: WheelNumber,
        var: sp.Symbol,
        value,
        trace: bool = False,
    ) -> WheelNumber:
        """
        'Limit' in wheel — simply substitutes the value.
        If it leads to /0 → ⊥ (instead of ∞).

        This is the philosophical core of the project:
        Wheel doesn't need limits — it substitutes and returns ⊥ instead of an error.
        """
        result = self.evaluate_at(expr, {var: value}, trace=trace)
        if trace:
            classical = "∞"
            try:
                classical = str(sp.limit(expr.value, var, value))
            except Exception:
                pass
            print(f"\n  Comparison:")
            print(f"  Classical limit lim({var}→{value}): {classical}")
            print(f"  Wheel substitution:                 {result}")
        return result

    # ── Step-by-step explanation ──────────────────────────────────────────────

    def explain(self, operation: str, *args) -> str:
        """Returns a text explanation of the operation."""
        lines = []
        op = operation.lower()

        if op in ("div", "/"):
            a, b = _coerce(args[0]), _coerce(args[1])
            lines.append(f"Calculating: {a} / {b}")
            lines.append(f"  Step 1: /({b}) = ?")
            b_inv = self.inv(b, trace=False)
            lines.append(f"         /({b}) = {b_inv}")
            lines.append(f"  Step 2: {a} * {b_inv} = ?")
            result = self.mul(a, b_inv)
            lines.append(f"         = {result}")
            if result.is_bottom:
                lines.append(f"  Result ⊥ — in classical algebra it would be: error or ∞")
            lines.append(f"  Final result: {result}")

        elif op in ("inv", "/x"):
            a = _coerce(args[0])
            lines.append(f"Calculating: /({a})")
            if a.is_bottom:
                lines.append(f"  /⊥ = ⊥  [⊥ is absorbing]")
            elif a.is_zero:
                lines.append(f"  /0 = ⊥  [axiom: 0·/0 = ⊥ in wheel]")
            else:
                lines.append(f"  /{a} = {a.wheel_inv()}  [standard inversion]")

        return "\n".join(lines)


# ─── Tests and demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    wa = WheelAlgebra()

    print("═" * 60)
    print("  WheelAlgebra — rules and rewriting")
    print("═" * 60)

    r, r_s = sp.symbols("r r_s", positive=True)

    # ── Test 1: Distributivity — where Wheel differs from classical algebra
    print("\n▶  Distributivity in Wheel vs classical algebra")
    print("\n  Case 1: x=2, y=3, z=4 (normal numbers)")
    wa.distributivity_check(W(2), W(3), W(4))

    print("\n  Case 2: x=0, y=1/0, z=0 (with ⊥)")
    wa.distributivity_check(W(0), W(BOTTOM), W(0))

    # ── Test 2: Wheel limit vs classical limit
    print("\n▶  Wheel 'limit' vs classical limit")
    print("\n  Schwarzschild g_rr: 1/(1 - r_s/r)")
    g_rr = W(1 / (1 - r_s / r))
    wa.wheel_limit(g_rr, r, r_s, trace=True)

    # ── Test 3: Step-by-step explanation
    print("\n▶  Step by step: 5 / 0")
    print(wa.explain("div", W(5), W(0)))

    print("\n▶  Step by step: 1 / (r - r_s) at r=r_s")
    expr_sym = W(1 / (r - r_s))
    result = wa.evaluate_at(expr_sym, {r: r_s}, trace=True)

    # ── Test 4: Key axiom — 0 * ⊥ = ⊥ (not 0!)
    print("\n▶  Axiom 10: 0 * ⊥ = ⊥  (not 0!)")
    result = wa.mul(W(0), W(BOTTOM), trace=True)
    print(f"  Classically: 0 * ∞ = NaN or error")
    print(f"  Wheel:       0 * ⊥ = {result}")

    print("\n" + "═" * 60)