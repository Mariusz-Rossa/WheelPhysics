# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
WheelNumber — Wheel Algebra implementation
Based on: Carlström (2004) "Wheels — On Division by Zero"

Wheel Algebra extends a ring with two new elements:
  - ⊥ (bottom/absurd) — result of undefined operations
  - /0 → ⊥ (instead of an error)

Wheel axioms:
  (1)  x + y = y + x
  (2)  (x + y) + z = x + (y + z)
  (3)  x + 0 = x
  (4)  x * y = y * x
  (5)  (x * y) * z = x * (y * z)
  (6)  x * 1 = x
  (7)  x * (y + z) = x*y + x*z  [!] NOT always — see ⊥ rule
  (8)  /(/x) = x
  (9)  x * /x = 1  (when x ≠ 0 and x ≠ ⊥)
  (10) 0 * /0 = ⊥  (key axiom)
  (11) ⊥ + x = ⊥
  (12) ⊥ * x = ⊥
  (13) /(x + y*⊥) = /x  [projection through ⊥]
"""

from __future__ import annotations
from typing import Union
import sympy as sp


# ─── Sentinel for the ⊥ element ───────────────────────────────────────────────

class _BottomType:
    """Singleton representing the ⊥ (bottom) element of wheel algebra."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "⊥"

    def __str__(self) -> str:
        return "⊥"

    def __eq__(self, other) -> bool:
        return isinstance(other, _BottomType)

    def __hash__(self) -> int:
        return hash("__wheel_bottom__")


BOTTOM = _BottomType()   # the sole instance of ⊥


# ─── WheelNumber ──────────────────────────────────────────────────────────────

WheelValue = Union[int, float, sp.Basic, _BottomType, "WheelNumber"]


class WheelNumber:
    """
    Wheel algebra element.

    The internal value is:
      - sp.Basic  (SymPy expression — symbolic or numeric)
      - BOTTOM    (the absurd element ⊥)

    Examples:
        >>> w = WheelNumber(3)
        >>> w / WheelNumber(0)
        WheelNumber(⊥)
        >>> WheelNumber(1) / WheelNumber(0) + WheelNumber(5)
        WheelNumber(⊥)
    """

    __slots__ = ("_val",)

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self, value: WheelValue = 0):
        if isinstance(value, WheelNumber):
            self._val = value._val
        elif isinstance(value, _BottomType):
            self._val = BOTTOM
        elif isinstance(value, (int, float)):
            self._val = sp.sympify(value)
        elif isinstance(value, sp.Basic):
            self._val = value
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_bottom(self) -> bool:
        return self._val is BOTTOM

    @property
    def is_zero(self) -> bool:
        if self.is_bottom:
            return False
        try:
            return bool(self._val == sp.S.Zero)
        except Exception:
            return False

    @property
    def value(self):
        return self._val

    # ── Representation ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"WheelNumber({self._val})"

    def __str__(self) -> str:
        return str(self._val)

    # ── Multiplicative inversion /x ──────────────────────────────────────────

    def wheel_inv(self) -> "WheelNumber":
        """
        Wheel inversion: /x
          /0   = ⊥     (axiom: 0 * /0 = ⊥)
          /⊥   = ⊥     (absurd propagates)
          /(/x) = x    (double inversion)
          /x   = 1/x   (for x ≠ 0, x ≠ ⊥)
        """
        if self.is_bottom:
            return WheelNumber(BOTTOM)
        if self.is_zero:
            return WheelNumber(BOTTOM)
        try:
            return WheelNumber(sp.Integer(1) / self._val)
        except Exception:
            return WheelNumber(BOTTOM)

    # ── Arithmetic operators ──────────────────────────────────────────────────

    def __add__(self, other: WheelValue) -> "WheelNumber":
        other = _coerce(other)
        # Axiom (11): ⊥ + x = ⊥
        if self.is_bottom or other.is_bottom:
            return WheelNumber(BOTTOM)
        try:
            result = sp.simplify(self._val + other._val)
            return WheelNumber(result)
        except Exception:
            return WheelNumber(BOTTOM)

    def __radd__(self, other: WheelValue) -> "WheelNumber":
        return _coerce(other).__add__(self)

    def __neg__(self) -> "WheelNumber":
        if self.is_bottom:
            return WheelNumber(BOTTOM)
        return WheelNumber(-self._val)

    def __sub__(self, other: WheelValue) -> "WheelNumber":
        return self.__add__(-_coerce(other))

    def __rsub__(self, other: WheelValue) -> "WheelNumber":
        return _coerce(other).__sub__(self)

    def __mul__(self, other: WheelValue) -> "WheelNumber":
        other = _coerce(other)
        # Axiom (12): ⊥ * x = ⊥
        if self.is_bottom or other.is_bottom:
            return WheelNumber(BOTTOM)
        try:
            result = sp.simplify(self._val * other._val)
            return WheelNumber(result)
        except Exception:
            return WheelNumber(BOTTOM)

    def __rmul__(self, other: WheelValue) -> "WheelNumber":
        return _coerce(other).__mul__(self)

    def __truediv__(self, other: WheelValue) -> "WheelNumber":
        """
        Wheel division: x / y = x * /y
        Key cases:
          x / 0  = x * /0 = x * ⊥ = ⊥   (for x ≠ 0)
          0 / 0  = 0 * /0 = 0 * ⊥ = ⊥   (axiom 10)
          x / ⊥  = ⊥
        """
        other = _coerce(other)
        return self.__mul__(other.wheel_inv())

    def __rtruediv__(self, other: WheelValue) -> "WheelNumber":
        return _coerce(other).__truediv__(self)

    def __pow__(self, exp: WheelValue) -> "WheelNumber":
        """
        Exponentiation — extension beyond standard axioms.
        x^0 = 1 (even for x=0, according to wheel convention)
        ⊥^n = ⊥
        """
        exp = _coerce(exp)
        if self.is_bottom:
            return WheelNumber(BOTTOM)
        if exp.is_bottom:
            return WheelNumber(BOTTOM)
        try:
            result = sp.simplify(self._val ** exp._val)
            # Check if SymPy returned infinity
            if result in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                return WheelNumber(BOTTOM)
            return WheelNumber(result)
        except Exception:
            return WheelNumber(BOTTOM)

    # ── Comparisons ───────────────────────────────────────────────────────────

    def __eq__(self, other) -> bool:
        other = _coerce(other)
        if self.is_bottom and other.is_bottom:
            return True
        if self.is_bottom or other.is_bottom:
            return False
        try:
            return bool(sp.simplify(self._val - other._val) == 0)
        except Exception:
            return False

    def __hash__(self) -> int:
        return hash(self._val)

    # ── Numeric conversion ────────────────────────────────────────────────────

    def to_float(self) -> float | None:
        """Attempt to convert to float. Returns None for ⊥."""
        if self.is_bottom:
            return None
        try:
            return float(self._val.evalf())
        except Exception:
            return None

    def evalf(self, n: int = 15) -> "WheelNumber":
        """Numeric evaluation (like SymPy .evalf())."""
        if self.is_bottom:
            return self
        try:
            result = self._val.evalf(n)
            if result in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                return WheelNumber(BOTTOM)
            return WheelNumber(result)
        except Exception:
            return WheelNumber(BOTTOM)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _coerce(val: WheelValue) -> WheelNumber:
    """Converts any value to a WheelNumber."""
    if isinstance(val, WheelNumber):
        return val
    return WheelNumber(val)


def W(value: WheelValue) -> WheelNumber:
    """Shortcut constructor — W(3), W(sp.Symbol('r')), W(BOTTOM)."""
    return WheelNumber(value)


# ─── Unit tests (run: python wheel_number.py) ────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  WheelNumber — axioms tests")
    print("=" * 60)

    r = sp.Symbol("r", positive=True)
    x = sp.Symbol("x")

    tests = [
        # (description, expression, expected result)
        ("1 / 0  →  ⊥",           W(1) / W(0),              W(BOTTOM)),
        ("0 / 0  →  ⊥",           W(0) / W(0),              W(BOTTOM)),
        ("⊥ + 5  →  ⊥",           W(BOTTOM) + W(5),         W(BOTTOM)),
        ("⊥ * 5  →  ⊥",           W(BOTTOM) * W(5),         W(BOTTOM)),
        ("1 / 1  →  1",           W(1) / W(1),              W(1)),
        ("6 / 2  →  3",           W(6) / W(2),              W(3)),
        ("/(W(0)) → ⊥",           W(0).wheel_inv(),         W(BOTTOM)),
        ("/(/3)  →  3",           W(3).wheel_inv().wheel_inv(), W(3)),
        ("2 + 3  →  5",           W(2) + W(3),              W(5)),
        ("r / 0  →  ⊥",           W(r) / W(0),              W(BOTTOM)),
        ("W(r) * W(1/r)  →  1",   W(r) * W(sp.Integer(1)/r), W(1)),
    ]

    passed = 0
    for desc, got, expected in tests:
        ok = got == expected
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        print(f"  {status}  {desc:<30}  got: {got}")

    print(f"\n  Result: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    sys.exit(0 if passed == len(tests) else 1)