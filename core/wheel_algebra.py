# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
wheel_algebra.py — reguły algebry koła i przepisywanie wyrażeń

Implementuje formalne reguły przepisywania wyrażeń w algebrze koła,
zgodnie z aksjomatami Carlströma (2004).

Kluczowe reguły:
  - Detekcja i propagacja ⊥
  - Normalizacja wyrażeń wheel
  - Redukcja form nieoznaczonych
  - Reguły dla granic w stylu wheel

Różnica od klasycznej algebry:
  Klasyczna:  x*(y+z) = x*y + x*z          (zawsze)
  Wheel:      x*(y+z) = x*y + x*z + 0*⊥   (gdy x, y, z ∉ {0, ⊥})
              — rozdzielność NIE zachodzi dla 0 i ⊥
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import sympy as sp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.wheel_number import WheelNumber, WheelValue, BOTTOM, W, _coerce


# ─── Reguły przepisywania ─────────────────────────────────────────────────────

@dataclass
class RewriteRule:
    """Pojedyncza reguła przepisywania z opisem."""
    name: str
    description: str
    apply: Callable[[WheelNumber, Optional[WheelNumber]], Optional[WheelNumber]]



def _contains_infinity(expr) -> bool:
    """Sprawdza czy wyrażenie zawiera zoo/oo/nan jako podwyrażenie."""
    if expr in (sp.oo, sp.zoo, sp.nan, -sp.oo):
        return True
    try:
        return expr.has(sp.oo) or expr.has(sp.zoo) or expr.has(sp.nan)
    except Exception:
        return False



def _has_division_by_zero_at(expr: sp.Basic, substitutions: dict) -> bool:
    """
    Sprawdza REKURENCYJNIE czy gdziekolwiek w wyrażeniu
    pojawia się dzielenie przez zero po podstawieniu.

    SymPy może algebraicznie uprościć 1/(1-r_s/r) przy r=0 do 0,
    tracąc informację o /r w środku. Ta funkcja chodzi po drzewie
    wyrażenia i sprawdza każdy mianownik przed uproszczeniem.
    """
    def walk(e) -> bool:
        # Sprawdź mianownik tego podwyrażenia
        n, d = sp.fraction(e)
        if d != sp.S.One:
            d_sub = sp.simplify(d.subs(substitutions))
            if d_sub == sp.S.Zero:
                return True
            if _contains_infinity(d_sub):
                return True

        # Rekurencja po argumentach
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
    System reguł algebry koła.

    Użycie:
        wa = WheelAlgebra()
        result = wa.simplify(expr)
        wa.explain(expr)  # krok po kroku
    """

    def __init__(self):
        self._rules = self._build_rules()

    # ── Budowa reguł ─────────────────────────────────────────────────────────

    def _build_rules(self) -> list[RewriteRule]:
        return [
            RewriteRule(
                name="bottom_absorb_add",
                description="⊥ + x = ⊥  (aksjomat 11)",
                apply=lambda a, b: W(BOTTOM) if (a.is_bottom or (b and b.is_bottom)) else None,
            ),
            RewriteRule(
                name="bottom_absorb_mul",
                description="⊥ * x = ⊥  (aksjomat 12)",
                apply=lambda a, b: W(BOTTOM) if (a.is_bottom or (b and b.is_bottom)) else None,
            ),
            RewriteRule(
                name="zero_inv",
                description="/0 = ⊥  (aksjomat 10: 0·/0 = ⊥)",
                apply=lambda a, b: W(BOTTOM) if a.is_zero else None,
            ),
            RewriteRule(
                name="double_inv",
                description="/(/x) = x  (aksjomat 8)",
                apply=lambda a, b: a,  # stosowane przez wheel_inv().wheel_inv()
            ),
            RewriteRule(
                name="bottom_inv",
                description="/⊥ = ⊥  (⊥ jest absorbujące)",
                apply=lambda a, b: W(BOTTOM) if a.is_bottom else None,
            ),
            RewriteRule(
                name="zero_mul_bottom",
                description="0 * ⊥ = ⊥  (nie 0!)",
                apply=lambda a, b: W(BOTTOM) if (
                    a.is_zero and b and b.is_bottom or
                    a.is_bottom and b and b.is_zero
                ) else None,
            ),
        ]

    # ── Operacje algebraiczne z śledzeniem ───────────────────────────────────

    def add(self, a: WheelValue, b: WheelValue, trace: bool = False) -> WheelNumber:
        a, b = _coerce(a), _coerce(b)
        if a.is_bottom or b.is_bottom:
            if trace:
                print(f"  ADD: ⊥ absorpcja → ⊥  [aksjomat 11]")
            return W(BOTTOM)
        result = a + b
        if trace:
            print(f"  ADD: {a} + {b} = {result}")
        return result

    def mul(self, a: WheelValue, b: WheelValue, trace: bool = False) -> WheelNumber:
        a, b = _coerce(a), _coerce(b)
        if a.is_bottom or b.is_bottom:
            if trace:
                print(f"  MUL: ⊥ absorpcja → ⊥  [aksjomat 12]")
            return W(BOTTOM)
        result = a * b
        if trace:
            print(f"  MUL: {a} * {b} = {result}")
        return result

    def inv(self, a: WheelValue, trace: bool = False) -> WheelNumber:
        """Inwersja multiplikatywna /a."""
        a = _coerce(a)
        if a.is_bottom:
            if trace:
                print(f"  INV: /⊥ = ⊥  [⊥ absorpcja]")
            return W(BOTTOM)
        if a.is_zero:
            if trace:
                print(f"  INV: /0 = ⊥  [aksjomat 10]")
            return W(BOTTOM)
        result = a.wheel_inv()
        if trace:
            print(f"  INV: /{a} = {result}")
        return result

    def div(self, a: WheelValue, b: WheelValue, trace: bool = False) -> WheelNumber:
        """Dzielenie: a/b = a * /b."""
        a, b = _coerce(a), _coerce(b)
        b_inv = self.inv(b, trace=trace)
        return self.mul(a, b_inv, trace=trace)

    # ── Normalizacja wyrażenia ────────────────────────────────────────────────

    def normalize(self, expr: WheelNumber) -> WheelNumber:
        """
        Próba uproszczenia wyrażenia wheel.
        Dla wyrażeń symbolicznych używa SymPy.simplify.
        """
        if expr.is_bottom:
            return expr
        try:
            simplified = sp.simplify(expr.value)
            # Sprawdź czy uproszenie nie wprowadziło nieskończoności
            if simplified in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                return W(BOTTOM)
            return W(simplified)
        except Exception:
            return expr

    # ── Ewaluacja wyrażenia przy podstawieniu ─────────────────────────────────

    def evaluate_at(
        self,
        expr: WheelNumber,
        substitutions: dict,
        trace: bool = False,
    ) -> WheelNumber:
        """
        Podstawia wartości do wyrażenia symbolicznego.
        Zwraca ⊥ jeśli podstawienie prowadzi do /0.

        Args:
            expr:          WheelNumber z symbolami SymPy
            substitutions: {symbol: wartość}, np. {r: 0}
            trace:         czy drukować kroki
        """
        if expr.is_bottom:
            return expr

        if trace:
            print(f"\n  Ewaluacja: {expr.value}")
            print(f"  Podstawienie: {substitutions}")

        try:
            # KLUCZOWE: sprawdź rekurencyjnie mianowniki PRZED podstawieniem
            # (SymPy może uprościć 1/(1-r_s/r) przy r=0 do 0, tracąc info o /r)
            if _has_division_by_zero_at(expr.value, substitutions):
                if trace:
                    print(f"  → Znaleziono /0 w podwyrażeniu → ⊥")
                return W(BOTTOM)

            substituted = expr.value.subs(substitutions)

            # Sprawdź czy wynik jest nieskończony / niezdefiniowany
            if substituted in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                if trace:
                    print(f"  → Klasyczny wynik: {substituted} → mapuję na ⊥")
                return W(BOTTOM)

            # Sprawdź czy wynik ZAWIERA zoo/oo jako czynnik (np. zoo*k)
            if _contains_infinity(substituted):
                if trace:
                    print(f"  → Zawiera nieskończoność ({substituted}) → ⊥")
                return W(BOTTOM)

            # Sprawdź mianowniki po podstawieniu
            numer, denom = sp.fraction(sp.simplify(substituted))
            if denom == sp.S.Zero or (hasattr(denom, 'is_zero') and denom.is_zero):
                if trace:
                    print(f"  → Mianownik = 0 → ⊥")
                return W(BOTTOM)

            result = W(substituted)
            if trace:
                print(f"  → Wynik: {result}")
            return result

        except (ZeroDivisionError, sp.core.sympify.SympifyError):
            if trace:
                print(f"  → Wyjątek dzielenia → ⊥")
            return W(BOTTOM)
        except Exception as e:
            if trace:
                print(f"  → Błąd: {e} → ⊥")
            return W(BOTTOM)

    # ── Rozdzielność — kluczowa różnica vs klasyczna algebra ─────────────────

    def distributivity_check(
        self,
        x: WheelValue,
        y: WheelValue,
        z: WheelValue,
        trace: bool = True,
    ) -> dict:
        """
        Sprawdza czy x*(y+z) = x*y + x*z w algebrze koła.

        W klasycznej algebrze to zawsze prawda.
        W wheel: NIE zachodzi gdy x=0 lub wynik zawiera ⊥.
        """
        x, y, z = _coerce(x), _coerce(y), _coerce(z)

        lhs = self.mul(x, self.add(y, z))         # x*(y+z)
        rhs = self.add(self.mul(x, y), self.mul(x, z))  # x*y + x*z

        holds = (lhs == rhs)

        if trace:
            print(f"\n  Rozdzielność: x*(y+z) = x*y + x*z ?")
            print(f"  x={x}, y={y}, z={z}")
            print(f"  LHS: {x}*({y}+{z}) = {x}*{self.add(y,z)} = {lhs}")
            print(f"  RHS: {x}*{y} + {x}*{z} = {self.mul(x,y)} + {self.mul(x,z)} = {rhs}")
            print(f"  Zachodzi: {'✓ TAK' if holds else '✗ NIE — to różni Wheel od klasycznej algebry!'}")

        return {"lhs": lhs, "rhs": rhs, "holds": holds}

    # ── Wheel limit — granica w stylu wheel ───────────────────────────────────

    def wheel_limit(
        self,
        expr: WheelNumber,
        var: sp.Symbol,
        value,
        trace: bool = False,
    ) -> WheelNumber:
        """
        'Granica' w wheel — po prostu podstawia wartość.
        Jeśli prowadzi do /0 → ⊥ (zamiast ∞).

        To jest filozoficzny rdzeń projektu:
        Wheel nie potrzebuje granic — podstawia i zwraca ⊥ zamiast błędu.
        """
        result = self.evaluate_at(expr, {var: value}, trace=trace)
        if trace:
            classical = "∞"
            try:
                classical = str(sp.limit(expr.value, var, value))
            except Exception:
                pass
            print(f"\n  Porównanie:")
            print(f"  Klasyczna granica lim({var}→{value}): {classical}")
            print(f"  Wheel podstawienie:                   {result}")
        return result

    # ── Wyjaśnienie krok po kroku ─────────────────────────────────────────────

    def explain(self, operation: str, *args) -> str:
        """Zwraca tekstowe wyjaśnienie operacji."""
        lines = []
        op = operation.lower()

        if op in ("div", "/"):
            a, b = _coerce(args[0]), _coerce(args[1])
            lines.append(f"Obliczam: {a} / {b}")
            lines.append(f"  Krok 1: /({b}) = ?")
            b_inv = self.inv(b, trace=False)
            lines.append(f"         /({b}) = {b_inv}")
            lines.append(f"  Krok 2: {a} * {b_inv} = ?")
            result = self.mul(a, b_inv)
            lines.append(f"         = {result}")
            if result.is_bottom:
                lines.append(f"  Wynik ⊥ — w klasycznej algebrze byłoby: błąd lub ∞")
            lines.append(f"  Wynik końcowy: {result}")

        elif op in ("inv", "/x"):
            a = _coerce(args[0])
            lines.append(f"Obliczam: /({a})")
            if a.is_bottom:
                lines.append(f"  /⊥ = ⊥  [⊥ jest absorbujące]")
            elif a.is_zero:
                lines.append(f"  /0 = ⊥  [aksjomat: 0·/0 = ⊥ w kole]")
            else:
                lines.append(f"  /{a} = {a.wheel_inv()}  [inwersja standardowa]")

        return "\n".join(lines)


# ─── Testy i demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    wa = WheelAlgebra()

    print("═" * 60)
    print("  WheelAlgebra — reguły i przepisywanie")
    print("═" * 60)

    r, r_s = sp.symbols("r r_s", positive=True)

    # ── Test 1: Rozdzielność — gdzie Wheel różni się od klasycznej algebry
    print("\n▶  Rozdzielność w Wheel vs klasyczna algebra")
    print("\n  Przypadek 1: x=2, y=3, z=4 (normalne liczby)")
    wa.distributivity_check(W(2), W(3), W(4))

    print("\n  Przypadek 2: x=0, y=1/0, z=0 (z ⊥)")
    wa.distributivity_check(W(0), W(BOTTOM), W(0))

    # ── Test 2: Wheel limit vs klasyczna granica
    print("\n▶  Wheel 'granica' vs klasyczna granica")
    print("\n  g_rr Schwarzschilda: 1/(1 - r_s/r)")
    g_rr = W(1 / (1 - r_s / r))
    wa.wheel_limit(g_rr, r, r_s, trace=True)

    # ── Test 3: Wyjaśnienie krok po kroku
    print("\n▶  Krok po kroku: 5 / 0")
    print(wa.explain("div", W(5), W(0)))

    print("\n▶  Krok po kroku: 1 / (r - r_s) przy r=r_s")
    expr_sym = W(1 / (r - r_s))
    result = wa.evaluate_at(expr_sym, {r: r_s}, trace=True)

    # ── Test 4: Kluczowy aksjomat — 0 * ⊥ = ⊥ (nie 0!)
    print("\n▶  Aksjomat 10: 0 * ⊥ = ⊥  (nie 0!)")
    result = wa.mul(W(0), W(BOTTOM), trace=True)
    print(f"  Klasycznie: 0 * ∞ = NaN lub błąd")
    print(f"  Wheel:      0 * ⊥ = {result}")

    print("\n" + "═" * 60)