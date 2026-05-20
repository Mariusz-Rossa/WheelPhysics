# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
WheelNumber — implementacja algebry koła (Wheel Algebra)
Oparty na: Carlström (2004) "Wheels — On Division by Zero"

Wheel Algebra rozszerza pierścień o dwa nowe elementy:
  - ⊥ (bottom/absurd) — wynik operacji niezdefiniowanych
  - /0 → ⊥ (zamiast błędu)

Aksjomaty koła:
  (1)  x + y = y + x
  (2)  (x + y) + z = x + (y + z)
  (3)  x + 0 = x
  (4)  x * y = y * x
  (5)  (x * y) * z = x * (y * z)
  (6)  x * 1 = x
  (7)  x * (y + z) = x*y + x*z  [!] NIE zawsze — patrz reguła ⊥
  (8)  /(/x) = x
  (9)  x * /x = 1  (gdy x ≠ 0 i x ≠ ⊥)
  (10) 0 * /0 = ⊥  (kluczowy aksjomat)
  (11) ⊥ + x = ⊥
  (12) ⊥ * x = ⊥
  (13) /(x + y*⊥) = /x  [rzut przez ⊥]
"""

from __future__ import annotations
from typing import Union
import sympy as sp


# ─── Sentinel dla elementu ⊥ ──────────────────────────────────────────────────

class _BottomType:
    """Singleton reprezentujący element ⊥ (bottom) algebry koła."""
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


BOTTOM = _BottomType()   # jedyna instancja ⊥


# ─── WheelNumber ──────────────────────────────────────────────────────────────

WheelValue = Union[int, float, sp.Basic, _BottomType, "WheelNumber"]


class WheelNumber:
    """
    Element algebry koła.

    Wartość wewnętrzna to:
      - sp.Basic  (wyrażenie SymPy — symboliczne lub numeryczne)
      - BOTTOM    (element absurdalny ⊥)

    Przykłady:
        >>> w = WheelNumber(3)
        >>> w / WheelNumber(0)
        WheelNumber(⊥)
        >>> WheelNumber(1) / WheelNumber(0) + WheelNumber(5)
        WheelNumber(⊥)
    """

    __slots__ = ("_val",)

    # ── Konstruktor ──────────────────────────────────────────────────────────

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
            raise TypeError(f"Nieobsługiwany typ: {type(value)}")

    # ── Właściwości ──────────────────────────────────────────────────────────

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

    # ── Reprezentacja ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"WheelNumber({self._val})"

    def __str__(self) -> str:
        return str(self._val)

    # ── Inwersja multiplikatywna /x ──────────────────────────────────────────

    def wheel_inv(self) -> "WheelNumber":
        """
        Inwersja koła: /x
          /0   = ⊥     (aksjomat: 0 * /0 = ⊥)
          /⊥   = ⊥     (absurd propaguje)
          /(/x) = x    (podwójna inwersja)
          /x   = 1/x   (dla x ≠ 0, x ≠ ⊥)
        """
        if self.is_bottom:
            return WheelNumber(BOTTOM)
        if self.is_zero:
            return WheelNumber(BOTTOM)
        try:
            return WheelNumber(sp.Integer(1) / self._val)
        except Exception:
            return WheelNumber(BOTTOM)

    # ── Operatory arytmetyczne ────────────────────────────────────────────────

    def __add__(self, other: WheelValue) -> "WheelNumber":
        other = _coerce(other)
        # Aksjomat (11): ⊥ + x = ⊥
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
        # Aksjomat (12): ⊥ * x = ⊥
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
        Dzielenie w kole: x / y = x * /y
        Kluczowe przypadki:
          x / 0  = x * /0 = x * ⊥ = ⊥   (dla x ≠ 0)
          0 / 0  = 0 * /0 = 0 * ⊥ = ⊥   (aksjomat 10)
          x / ⊥  = ⊥
        """
        other = _coerce(other)
        return self.__mul__(other.wheel_inv())

    def __rtruediv__(self, other: WheelValue) -> "WheelNumber":
        return _coerce(other).__truediv__(self)

    def __pow__(self, exp: WheelValue) -> "WheelNumber":
        """
        Potęgowanie — rozszerzenie poza standardowy aksjomat.
        x^0 = 1 (nawet dla x=0, zgodnie z konwencją koła)
        ⊥^n = ⊥
        """
        exp = _coerce(exp)
        if self.is_bottom:
            return WheelNumber(BOTTOM)
        if exp.is_bottom:
            return WheelNumber(BOTTOM)
        try:
            result = sp.simplify(self._val ** exp._val)
            # Sprawdź czy SymPy zwrócił nieskończoność
            if result in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                return WheelNumber(BOTTOM)
            return WheelNumber(result)
        except Exception:
            return WheelNumber(BOTTOM)

    # ── Porównania ────────────────────────────────────────────────────────────

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

    # ── Konwersja numeryczna ──────────────────────────────────────────────────

    def to_float(self) -> float | None:
        """Próba konwersji do float. Zwraca None dla ⊥."""
        if self.is_bottom:
            return None
        try:
            return float(self._val.evalf())
        except Exception:
            return None

    def evalf(self, n: int = 15) -> "WheelNumber":
        """Numeryczna ewaluacja (jak SymPy .evalf())."""
        if self.is_bottom:
            return self
        try:
            result = self._val.evalf(n)
            if result in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                return WheelNumber(BOTTOM)
            return WheelNumber(result)
        except Exception:
            return WheelNumber(BOTTOM)


# ─── Pomocnicze ───────────────────────────────────────────────────────────────

def _coerce(val: WheelValue) -> WheelNumber:
    """Konwertuje dowolną wartość do WheelNumber."""
    if isinstance(val, WheelNumber):
        return val
    return WheelNumber(val)


def W(value: WheelValue) -> WheelNumber:
    """Skrócony konstruktor — W(3), W(sp.Symbol('r')), W(BOTTOM)."""
    return WheelNumber(value)


# ─── Testy jednostkowe (uruchom: python wheel_number.py) ─────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  WheelNumber — testy aksjomatów")
    print("=" * 60)

    r = sp.Symbol("r", positive=True)
    x = sp.Symbol("x")

    tests = [
        # (opis, wyrażenie, oczekiwany wynik)
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

    print(f"\n  Wynik: {passed}/{len(tests)} testów przeszło")
    print("=" * 60)
    sys.exit(0 if passed == len(tests) else 1)