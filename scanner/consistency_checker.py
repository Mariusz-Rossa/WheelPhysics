# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
consistency_checker.py — weryfikacja spójności tłumaczeń Wheel

Sprawdza czy przepisanie wyrażenia klasycznego → Wheel:
  1. Zachowuje wyniki tam gdzie oba systemy powinny się zgadzać
  2. Poprawnie identyfikuje miejsca rozbieżności (osobliwości)
  3. Nie wprowadza fałszywych ⊥ (false positives)
  4. Nie przegapia prawdziwych osobliwości (false negatives)

Metryki spójności:
  - Zgodność w punktach regularnych (powinno być 100%)
  - Pokrycie osobliwości (ile znanych osobliwości wykryto)
  - False positive rate (ile ⊥ jest niepoprawnych)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Optional
import sympy as sp

from core.wheel_number import W, BOTTOM
from core.wheel_algebra import WheelAlgebra
from core.sympy_extension import wheel_subs, is_singular_at
from core.wheel_calculus import (
    wheel_limit, RemovableSingularity, PoleSingularity,
    classify_batch, analyse_singularity,
)

_wa = WheelAlgebra()


# ─── Wynik weryfikacji ────────────────────────────────────────────────────────

@dataclass
class ConsistencyReport:
    expression:       sp.Basic
    name:             str
    regular_checks:   list[dict] = field(default_factory=list)
    singular_checks:  list[dict] = field(default_factory=list)
    false_positives:  list[dict] = field(default_factory=list)
    false_negatives:  list[dict] = field(default_factory=list)

    @property
    def regular_ok(self) -> int:
        return sum(1 for c in self.regular_checks if c["match"])

    @property
    def regular_total(self) -> int:
        return len(self.regular_checks)

    @property
    def singular_ok(self) -> int:
        return sum(1 for c in self.singular_checks if c["wheel_is_bottom"])

    @property
    def singular_total(self) -> int:
        return len(self.singular_checks)

    def summary(self) -> str:
        lines = [
            "─" * 62,
            f"WYRAŻENIE: {self.name}  [{self.expression}]",
            "",
            f"Punkty regularne  : {self.regular_ok}/{self.regular_total} zgodnych",
        ]

        for c in self.regular_checks:
            mark = "✓" if c["match"] else "✗"
            lines.append(
                f"  {mark}  {c['var']}={c['value']:<12} "
                f"klasycznie={str(c['classical']):<15} "
                f"Wheel={c['wheel']}"
            )

        lines.append(f"\nPunkty osobliwe   : {self.singular_ok}/{self.singular_total} wykrytych jako ⊥")
        for c in self.singular_checks:
            mark = "✓" if c["wheel_is_bottom"] else "✗ PRZEOCZONO"
            lines.append(
                f"  {mark}  {c['var']}={c['value']:<12} "
                f"Wheel={c['wheel']}"
            )

        if self.false_positives:
            lines.append(f"\n⚠  Fałszywe ⊥ (false positives): {len(self.false_positives)}")
            for fp in self.false_positives:
                lines.append(f"   {fp['var']}={fp['value']} — Wheel dał ⊥, ale punkt jest regularny")

        if self.false_negatives:
            lines.append(f"\n⚠  Przeoczone osobliwości (false negatives): {len(self.false_negatives)}")
            for fn in self.false_negatives:
                lines.append(f"   {fn['var']}={fn['value']} — osobliwość nie wykryta!")

        score = self._score()
        lines.append(f"\nOcena spójności   : {score:.0%}")
        lines.append("─" * 62)
        return "\n".join(lines)

    def _score(self) -> float:
        total = self.regular_total + self.singular_total
        if total == 0:
            return 1.0
        correct = self.regular_ok + self.singular_ok
        penalty = len(self.false_positives) + len(self.false_negatives)
        return max(0.0, (correct - penalty) / total)


# ─── Checker ──────────────────────────────────────────────────────────────────

class ConsistencyChecker:
    """
    Weryfikuje spójność tłumaczenia klasyczne → Wheel.

    Użycie:
        checker = ConsistencyChecker()
        report  = checker.check(
            expr,
            var=r,
            regular_values=[2*r_s, 3*r_s],
            singular_values=[r_s, 0],
            known_singular=[r_s, 0],
        )
        print(report.summary())
    """

    def check(
        self,
        expr:             sp.Basic,
        var:              sp.Symbol,
        regular_values:   list,
        singular_values:  list,
        known_singular:   Optional[list] = None,
        name:             str = "",
        extra_subs:       Optional[dict] = None,
    ) -> ConsistencyReport:
        """
        Weryfikuje wyrażenie w podanych punktach.

        Args:
            expr:            wyrażenie SymPy
            var:             główna zmienna
            regular_values:  wartości gdzie powinno być skończone
            singular_values: wartości gdzie spodziewamy się ⊥
            known_singular:  lista znanych osobliwości (do sprawdzenia pokrycia)
            name:            nazwa wyrażenia
            extra_subs:      dodatkowe podstawienia (np. {r_s: 1})
        """
        extra = extra_subs or {}
        report = ConsistencyReport(expression=expr, name=name or str(expr))

        # ── Punkty regularne
        for val in regular_values:
            subs = {var: val, **extra}
            wheel_result = wheel_subs(expr, subs)

            # Klasyczny wynik
            classical = self._classical_eval(expr, subs)

            # Porównanie (jeśli oba skończone)
            match = self._results_match(wheel_result, classical)

            report.regular_checks.append({
                "var": str(var), "value": str(val),
                "wheel": wheel_result,
                "classical": classical,
                "match": match,
            })

            # False positive: Wheel dał ⊥ w regularnym punkcie
            if wheel_result.is_bottom and not self._is_classically_singular(classical):
                report.false_positives.append({"var": str(var), "value": str(val)})

        # ── Punkty osobliwe
        for val in singular_values:
            subs = {var: val, **extra}
            wheel_result = wheel_subs(expr, subs)
            is_bottom = wheel_result.is_bottom

            report.singular_checks.append({
                "var": str(var), "value": str(val),
                "wheel": wheel_result,
                "wheel_is_bottom": is_bottom,
            })

        # ── False negatives (znane osobliwości które Wheel przeoczył)
        if known_singular:
            for val in known_singular:
                subs = {var: val, **extra}
                wheel_result = wheel_subs(expr, subs)
                if not wheel_result.is_bottom:
                    report.false_negatives.append({"var": str(var), "value": str(val)})

        return report

    def _classical_eval(self, expr: sp.Basic, subs: dict):
        """Próba klasycznej ewaluacji — zwraca wynik lub string."""
        try:
            result = expr.subs(subs)
            simplified = sp.simplify(result)
            return simplified
        except Exception:
            return "błąd"

    def _is_classically_singular(self, classical_result) -> bool:
        """Czy klasyczny wynik jest osobliwy (nieskończony/niezdefiniowany)?"""
        if isinstance(classical_result, str):
            return True
        try:
            return classical_result in (sp.oo, sp.zoo, sp.nan, -sp.oo) or \
                   classical_result.has(sp.oo) or classical_result.has(sp.zoo)
        except Exception:
            return False

    def _results_match(self, wheel_result, classical_result) -> bool:
        """Sprawdza czy wyniki Wheel i klasyczny są zgodne."""
        if wheel_result.is_bottom:
            return self._is_classically_singular(classical_result)
        if self._is_classically_singular(classical_result):
            return False
        try:
            diff = sp.simplify(wheel_result.value - classical_result)
            return bool(diff == 0)
        except Exception:
            return str(wheel_result) == str(classical_result)

    def run_calculus_suite(self, max_order: int = 8) -> dict:
        """
        Testuje wheel_calculus na kontrprzykładach z bazy równań.

        Weryfikuje trójpodział:
          REMOVABLE — Wheel=⊥ ale granica analityczna istnieje
          BOTTOM    — prawdziwy biegun (⊥ nieusuwalna)
          FINITE    — punkt regularny (Wheel OK bez ingerencji)

        Returns:
            słownik ze statystykami: total, correct, removable, poles, finite
        """
        print("\n" + "═" * 62)
        print("  WHEEL CALCULUS — Weryfikacja czwórpodziału osobliwości")
        print("  wheel_algebra (⊥) vs wheel_calculus (⊥/lim/rząd+res)")
        print("═" * 62)

        x, m, p, r, r_s = sp.symbols("x m p r r_s", real=True)
        omega, omega0    = sp.symbols("omega omega0", real=True)

        # ── Kontrprzykłady z bazy — powinny być USUWALNE ─────────────────
        print("\n  [1/3] Kontrprzykłady z bazy (Wheel=⊥, ale granica istnieje)\n")
        cases_removable = [
            {
                "name": "sinc(x) = sin(x)/x",
                "expr": sp.sin(x) / x,
                "variables": [(x, sp.S.Zero)],
                "expected_limit": sp.S.One,
            },
            {
                "name": "sinc²(x) = sin²(x)/x²",
                "expr": sp.sin(x)**2 / x**2,
                "variables": [(x, sp.S.Zero)],
                "expected_limit": sp.S.One,
            },
            {
                "name": "(1 - cos(x))/x²  [lim=1/2, nie 1!]",
                "expr": (1 - sp.cos(x)) / x**2,
                "variables": [(x, sp.S.Zero)],
                "expected_limit": sp.Rational(1, 2),
            },
            {
                "name": "Rayleigh-Jeans: ħω/(e^(ħω/kT)-1) gdy T→∞",
                "expr": x / (sp.exp(x) - 1),           # x = ħω/kT → 0
                "variables": [(x, sp.S.Zero)],
                "expected_limit": sp.S.One,
            },
        ]

        print(f"  {'Równanie':<44} {'Typ':<12} {'lim':<8} {'OK?'}")
        print(f"  {'─'*44} {'─'*12} {'─'*8} {'─'*4}")
        res_rem = classify_batch(cases_removable, max_order=max_order, verbose=True)
        rem_ok  = sum(1 for r in res_rem if r["correct"] and r["result_type"] == "REMOVABLE")

        # ── Prawdziwe bieguny — powinny być POLE z residuum ──────────────
        print("\n  [2/3] Prawdziwe bieguny z bazy (⊥ — teraz z rządem i residuum)\n")
        cases_poles = [
            {
                "name": "Propagator skalarny 1/(p²-m²) [on-shell]",
                "expr": 1 / (p**2 - m**2),
                "variables": [(p, m)],
                "expected_limit": sp.zoo,
            },
            {
                "name": "g_rr Schwarzschilda przy r=r_s",
                "expr": 1 / (1 - r_s / r),
                "variables": [(r, r_s)],
                "expected_limit": sp.zoo,
            },
            {
                "name": "Rezonans 1/(ω²-ω₀²) przy ω=ω₀",
                "expr": 1 / (omega**2 - omega0**2),
                "variables": [(omega, omega0)],
                "expected_limit": sp.zoo,
            },
            {
                "name": "Propagator fotonowy 1/k² przy k=0",
                "expr": 1 / x**2,
                "variables": [(x, sp.S.Zero)],
                "expected_limit": sp.zoo,
            },
        ]

        print(f"  {'Równanie':<44} {'Typ':<14} {'res':<16} {'OK?'}")
        print(f"  {'─'*44} {'─'*14} {'─'*16} {'─'*4}")
        res_pol = classify_batch(cases_poles, max_order=max_order, verbose=True)
        pol_ok  = sum(1 for r in res_pol if r["correct"] and r["result_type"] in ("POLE", "BOTTOM"))

        # ── Punkty regularne — Wheel nie powinien ingerować ───────────────
        print("\n  [3/3] Punkty regularne (Wheel OK, calculus nie ingeruje)\n")
        cases_reg = [
            {
                "name": "KG euklidesowy 1/(p²+m²) przy p=0, m=1",
                "expr": 1 / (p**2 + m**2),
                "variables": [(p, sp.S.Zero), (m, sp.S.One)],
                "expected_limit": sp.S.One,
            },
            {
                "name": "Potencjał efektywny fotonu przy r=3r_s/2",
                "expr": (1 - r_s / r) / r**2,
                "variables": [(r, sp.Rational(3, 2) * r_s)],
                "expected_limit": None,  # skończony, ale symboliczny
            },
        ]

        print(f"  {'Równanie':<44} {'Typ':<12} {'lim':<8} {'OK?'}")
        print(f"  {'─'*44} {'─'*12} {'─'*8} {'─'*4}")
        res_reg = classify_batch(cases_reg, max_order=max_order, verbose=True)
        reg_ok  = sum(1 for r in res_reg if r["result_type"] == "FINITE")

        # ── Podsumowanie ─────────────────────────────────────────────────
        total    = len(res_rem) + len(res_pol) + len(res_reg)
        correct  = rem_ok + pol_ok + reg_ok
        score    = correct / total if total else 0.0

        print(f"\n{'─'*62}")
        print(f"  Wynik wheel_calculus:")
        print(f"    Osobliwości usuwalne  : {rem_ok}/{len(res_rem)} poprawnie → RemovableSingularity")
        print(f"    Bieguny z residuum    : {pol_ok}/{len(res_pol)} poprawnie → PoleSingularity")
        print(f"    Punkty regularne      : {reg_ok}/{len(res_reg)} poprawnie → skończone")
        print(f"    Łącznie               : {correct}/{total}  ({score:.0%})")
        print(f"{'═'*62}")

        # ── Kluczowy przykład z raportem ─────────────────────────────────
        print("\n  Szczegółowy raport — sinc(x) (kluczowy kontrprzykład):\n")
        analyse_singularity(
            sp.sin(x) / x,
            [(x, sp.S.Zero)],
            name="sinc(x) = sin(x)/x",
            max_order=6,
            verbose=True,
        )

        return {
            "total": total, "correct": correct, "score": score,
            "removable": rem_ok, "poles": pol_ok, "finite": reg_ok,
        }

    def run_standard_suite(self) -> None:
        """Uruchamia standardowy zestaw testów spójności dla znanych równań."""

        print("═" * 62)
        print("  CONSISTENCY CHECKER — Standardowy zestaw testów")
        print("═" * 62)

        r, r_s, m, p, a, k_sym = sp.symbols("r r_s m p a k")

        tests = [
            {
                "name":           "g_rr Schwarzschilda",
                "expr":           1 / (1 - r_s / r),
                "var":            r,
                "extra_subs":     {r_s: sp.Integer(1)},
                "regular_values": [sp.Rational(2), sp.Rational(3), sp.Rational(5)],
                "singular_values":[sp.Integer(1), sp.Integer(0)],
                "known_singular": [sp.Integer(1)],
            },
            {
                "name":           "Niezmiennik Kretschmanna K=12r_s²/r⁶",
                "expr":           12 * r_s**2 / r**6,
                "var":            r,
                "extra_subs":     {r_s: sp.Integer(1)},
                "regular_values": [sp.Integer(1), sp.Rational(3, 2), sp.Integer(2)],
                "singular_values":[sp.Integer(0)],
                "known_singular": [sp.Integer(0)],
            },
            {
                "name":           "Propagator skalarny 1/(p²-m²)",
                "expr":           1 / (p**2 - m**2),
                "var":            p,
                "extra_subs":     {m: sp.Integer(1)},
                "regular_values": [sp.Integer(0), sp.Rational(1, 2), sp.Rational(3, 2)],
                "singular_values":[sp.Integer(1), sp.Integer(-1)],
                "known_singular": [sp.Integer(1), sp.Integer(-1)],
            },
            {
                "name":           "Człon Friedmanna k·c²/a² (k=c=1)",
                "expr":           k_sym / a**2,
                "var":            a,
                "extra_subs":     {k_sym: sp.Integer(1)},
                "regular_values": [sp.Integer(1), sp.Integer(2), sp.Rational(1, 2)],
                "singular_values":[sp.Integer(0)],
                "known_singular": [sp.Integer(0)],
            },
            {
                "name":           "Wyrażenie bez osobliwości: x²+1",
                "expr":           r**2 + 1,
                "var":            r,
                "extra_subs":     {},
                "regular_values": [sp.Integer(0), sp.Integer(1), sp.Integer(-1)],
                "singular_values":[],
                "known_singular": [],
            },
        ]

        total_score = 0.0
        for test in tests:
            report = self.check(
                expr=test["expr"],
                var=test["var"],
                regular_values=test["regular_values"],
                singular_values=test["singular_values"],
                known_singular=test["known_singular"],
                name=test["name"],
                extra_subs=test["extra_subs"],
            )
            print(f"\n{report.summary()}")
            total_score += report._score()

        avg = total_score / len(tests)
        print(f"\n{'═'*62}")
        print(f"  WYNIK GLOBALNY: {avg:.0%}  ({len(tests)} wyrażeń)")
        print(f"{'═'*62}")


if __name__ == "__main__":
    checker = ConsistencyChecker()
    checker.run_standard_suite()