# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
viz/comparator.py — porównanie wizualne: klasyczna vs Wheel Algebra

Generuje wykresy tekstowe (ASCII) i raporty pokazujące:
  - jak wyrażenie zachowuje się klasycznie vs w Wheel
  - gdzie pojawiają się ⊥ (oznaczone pionowo)
  - porównanie dla wielu wyrażeń jednocześnie
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
import sympy as sp

from core.wheel_number import W, BOTTOM
from core.sympy_extension import wheel_subs


# ─── Wykres ASCII ─────────────────────────────────────────────────────────────

class ASCIIPlot:
    """
    Prosty wykres ASCII dla funkcji jednej zmiennej.
    Oznacza ⊥ pionową linią '|⊥|'.
    """

    def __init__(self, width: int = 60, height: int = 12):
        self.width  = width
        self.height = height

    def plot(
        self,
        expr:       sp.Basic,
        var:        sp.Symbol,
        x_range:    tuple,
        extra_subs: Optional[dict] = None,
        title:      str = "",
        n_points:   int = 40,
    ) -> str:
        """
        Rysuje wykres ASCII wyrażenia.

        Zwraca string z wykresem.
        """
        extra = extra_subs or {}
        x_min, x_max = float(x_range[0]), float(x_range[1])
        step = (x_max - x_min) / n_points

        # Zbierz wartości
        xs, ys_classical, ys_wheel = [], [], []
        bottom_positions = []

        for i in range(n_points + 1):
            x_val = x_min + i * step
            xs.append(x_val)

            subs = {var: sp.Float(x_val), **extra}

            # Wheel
            w = wheel_subs(expr, subs)
            if w.is_bottom:
                ys_wheel.append(None)
                bottom_positions.append(i)
            else:
                try:
                    ys_wheel.append(float(w.value.evalf()))
                except Exception:
                    ys_wheel.append(None)

            # Klasyczny
            try:
                classical = float(sp.sympify(expr).subs(subs).evalf())
                if abs(classical) > 1e6 or classical != classical:
                    ys_classical.append(None)
                else:
                    ys_classical.append(classical)
            except Exception:
                ys_classical.append(None)

        # Znajdź zakres y
        valid = [y for y in ys_wheel if y is not None and abs(y) < 1e6]
        if not valid:
            valid = [y for y in ys_classical if y is not None]
        if not valid:
            return f"[Brak danych do wykresu dla {expr}]"

        y_min = min(valid) * 1.1 if min(valid) < 0 else min(valid) * 0.9
        y_max = max(valid) * 1.1 if max(valid) > 0 else max(valid) * 0.9
        if y_min == y_max:
            y_min -= 1; y_max += 1

        # Buduj siatkę
        grid = [[" "] * (n_points + 1) for _ in range(self.height)]

        def y_to_row(y_val):
            if y_val is None:
                return None
            row = int((y_max - y_val) / (y_max - y_min) * (self.height - 1))
            return max(0, min(self.height - 1, row))

        # Narysuj wartości wheel (·) i klasyczne (o)
        for i, (yw, yc) in enumerate(zip(ys_wheel, ys_classical)):
            if i in bottom_positions:
                # Kolumna ⊥ — pionowa linia
                for row in range(self.height):
                    grid[row][i] = "│"
            else:
                row_w = y_to_row(yw)
                if row_w is not None:
                    grid[row_w][i] = "·"

        # Oś zerowa
        zero_row = y_to_row(0.0)
        if zero_row is not None:
            for col in range(n_points + 1):
                if grid[zero_row][col] == " ":
                    grid[zero_row][col] = "─"

        # Złóż string
        lines = []
        if title:
            lines.append(f"  {title}")
            lines.append(f"  {'─' * len(title)}")

        y_labels = [f"{y_max:.2f}", "", "", "", "", f"{(y_max+y_min)/2:.2f}", "", "", "", "", "", f"{y_min:.2f}"]
        for row_i, row in enumerate(grid):
            label = y_labels[row_i] if row_i < len(y_labels) else ""
            lines.append(f"  {label:>8} │{''.join(row)}")

        # Oś X
        x_axis = "─" * (n_points + 1)
        lines.append(f"  {'':>8} └{x_axis}")
        lines.append(f"  {'':>9}{x_min:<15.2f}{'':>10}{(x_min+x_max)/2:<10.2f}{x_max:>8.2f}")

        # Legenda
        bottom_x = [f"{x_min + bp * step:.2f}" for bp in bottom_positions[:3]]
        if bottom_x:
            lines.append(f"\n  · = Wheel   │ = ⊥ (osobliwość)   ⊥ przy x ≈ {', '.join(bottom_x)}")
        else:
            lines.append(f"\n  · = Wheel   (brak osobliwości w zakresie)")

        return "\n".join(lines)


# ─── Comparator ───────────────────────────────────────────────────────────────

class Comparator:
    """
    Porównuje zachowanie wyrażeń w klasycznej algebrze i Wheel.
    Generuje raporty tekstowe z wykresami ASCII.
    """

    def __init__(self):
        self._plotter = ASCIIPlot()

    def compare(
        self,
        expr:       sp.Basic,
        var:        sp.Symbol,
        x_range:    tuple,
        extra_subs: Optional[dict] = None,
        name:       str = "",
        test_points: Optional[list] = None,
    ) -> str:
        """Generuje pełny raport porównawczy."""

        extra  = extra_subs or {}
        lines  = []
        header = name or str(expr)

        lines.append("═" * 64)
        lines.append(f"  PORÓWNANIE: {header}")
        lines.append("═" * 64)
        lines.append(f"  Wyrażenie : {expr}")
        lines.append(f"  Zmienna   : {var}  ∈  [{x_range[0]}, {x_range[1]}]")
        if extra:
            lines.append(f"  Parametry : {extra}")

        # Tabela wartości w punktach testowych
        if test_points:
            lines.append(f"\n  {'':─<60}")
            lines.append(f"  {'Punkt':<15} {'Klasycznie':<20} {'Wheel':<15} {'Status'}")
            lines.append(f"  {'':─<60}")

            for pt in test_points:
                subs = {var: pt, **extra}
                w = wheel_subs(expr, subs)

                try:
                    classical = sp.simplify(sp.sympify(expr).subs(subs))
                    cl_str = str(classical)[:18]
                except Exception:
                    cl_str = "błąd"

                if w.is_bottom:
                    w_str  = "⊥"
                    status = "← OSOBLIWOŚĆ"
                else:
                    try:
                        w_str = str(w.value)[:14]
                    except Exception:
                        w_str = str(w)
                    status = "✓ zgodne" if cl_str.replace(" ", "") == w_str.replace(" ", "") else "≈ różne"

                lines.append(f"  {str(pt):<15} {cl_str:<20} {w_str:<15} {status}")

        # Wykres ASCII
        lines.append(f"\n  Wykres (·=Wheel, │=⊥):\n")
        plot = self._plotter.plot(
            expr, var, x_range, extra_subs=extra, n_points=50
        )
        lines.append(plot)
        lines.append("═" * 64)
        return "\n".join(lines)

    def run_showcase(self) -> None:
        """Pokazuje porównania dla kluczowych równań projektu."""

        r, r_s, p, m, a = sp.symbols("r r_s p m a", real=True)

        print("█" * 64)
        print("  WHEELPHYSICS — Wizualne porównania Klasyczna vs Wheel")
        print("█" * 64)

        # ── g_rr Schwarzschilda
        print("\n" + self.compare(
            expr=1 / (1 - r_s / r),
            var=r,
            x_range=(0.1, 4.0),
            extra_subs={r_s: sp.Integer(1)},
            name="g_rr Schwarzschilda  (r_s=1)",
            test_points=[
                sp.Rational(1, 2), sp.Integer(1),
                sp.Rational(3, 2), sp.Integer(2), sp.Integer(3),
            ],
        ))

        # ── Propagator skalarny
        print("\n" + self.compare(
            expr=1 / (p**2 - m**2),
            var=p,
            x_range=(-3.0, 3.0),
            extra_subs={m: sp.Integer(1)},
            name="Propagator skalarny 1/(p²-m²)  (m=1)",
            test_points=[
                sp.Integer(-2), sp.Integer(-1), sp.Rational(-1, 2),
                sp.Integer(0),  sp.Rational(1, 2), sp.Integer(1), sp.Integer(2),
            ],
        ))

        # ── Friedmann
        print("\n" + self.compare(
            expr=sp.Integer(1) / a**2,
            var=a,
            x_range=(-2.0, 2.0),
            extra_subs={},
            name="Człon Friedmanna 1/a²  (k=c=1)",
            test_points=[
                sp.Integer(-2), sp.Integer(-1), sp.Rational(-1, 2),
                sp.Integer(0),  sp.Rational(1, 2), sp.Integer(1),
            ],
        ))


if __name__ == "__main__":
    Comparator().run_showcase()