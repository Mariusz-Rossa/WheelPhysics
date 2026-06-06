# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
consistency_checker.py — consistency verification of Wheel translations

Checks if the expression rewriting from classical → Wheel:
  1. Preserves results where both systems should agree
  2. Correctly identifies points of divergence (singularities)
  3. Does not introduce false ⊥ (false positives)
  4. Does not miss real singularities (false negatives)

Consistency metrics:
  - Agreement at regular points (should be 100%)
  - Singularity coverage (how many known singularities were detected)
  - False positive rate (how many ⊥ are incorrect)
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


# ─── Verification result ──────────────────────────────────────────────────────

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
            f"EXPRESSION: {self.name}  [{self.expression}]",
            "",
            f"Regular points    : {self.regular_ok}/{self.regular_total} matching",
        ]

        for c in self.regular_checks:
            mark = "✓" if c["match"] else "✗"
            lines.append(
                f"  {mark}  {c['var']}={c['value']:<12} "
                f"classical={str(c['classical']):<15} "
                f"Wheel={c['wheel']}"
            )

        lines.append(f"\nSingular points   : {self.singular_ok}/{self.singular_total} detected as ⊥")
        for c in self.singular_checks:
            mark = "✓" if c["wheel_is_bottom"] else "✗ MISSED"
            lines.append(
                f"  {mark}  {c['var']}={c['value']:<12} "
                f"Wheel={c['wheel']}"
            )

        if self.false_positives:
            lines.append(f"\n⚠  False ⊥ (false positives): {len(self.false_positives)}")
            for fp in self.false_positives:
                lines.append(f"   {fp['var']}={fp['value']} — Wheel gave ⊥, but the point is regular")

        if self.false_negatives:
            lines.append(f"\n⚠  Missed singularities (false negatives): {len(self.false_negatives)}")
            for fn in self.false_negatives:
                lines.append(f"   {fn['var']}={fn['value']} — singularity not detected!")

        score = self._score()
        lines.append(f"\nConsistency score : {score:.0%}")
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
    Verifies the consistency of classical → Wheel translation.

    Usage:
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
        Verifies an expression at specified points.

        Args:
            expr:            SymPy expression
            var:             main variable
            regular_values:  values where it should be finite
            singular_values: values where ⊥ is expected
            known_singular:  list of known singularities (to verify coverage)
            name:            expression name
            extra_subs:      additional substitutions (e.g., {r_s: 1})
        """
        extra = extra_subs or {}
        report = ConsistencyReport(expression=expr, name=name or str(expr))

        # ── Regular points
        for val in regular_values:
            subs = {var: val, **extra}
            wheel_result = wheel_subs(expr, subs)

            # Classical result
            classical = self._classical_eval(expr, subs)

            # Comparison (if both are finite)
            match = self._results_match(wheel_result, classical)

            report.regular_checks.append({
                "var": str(var), "value": str(val),
                "wheel": wheel_result,
                "classical": classical,
                "match": match,
            })

            # False positive: Wheel gave ⊥ at a regular point
            if wheel_result.is_bottom and not self._is_classically_singular(classical):
                report.false_positives.append({"var": str(var), "value": str(val)})

        # ── Singular points
        for val in singular_values:
            subs = {var: val, **extra}
            wheel_result = wheel_subs(expr, subs)
            is_bottom = wheel_result.is_bottom

            report.singular_checks.append({
                "var": str(var), "value": str(val),
                "wheel": wheel_result,
                "wheel_is_bottom": is_bottom,
            })

        # ── False negatives (known singularities that Wheel missed)
        if known_singular:
            for val in known_singular:
                subs = {var: val, **extra}
                wheel_result = wheel_subs(expr, subs)
                if not wheel_result.is_bottom:
                    report.false_negatives.append({"var": str(var), "value": str(val)})

        return report

    def _classical_eval(self, expr: sp.Basic, subs: dict):
        """Attempts classical evaluation — returns result or string."""
        try:
            result = expr.subs(subs)
            simplified = sp.simplify(result)
            return simplified
        except Exception:
            return "error"

    def _is_classically_singular(self, classical_result) -> bool:
        """Is the classical result singular (infinite/undefined)?"""
        if isinstance(classical_result, str):
            return True
        try:
            return classical_result in (sp.oo, sp.zoo, sp.nan, -sp.oo) or \
                   classical_result.has(sp.oo) or classical_result.has(sp.zoo)
        except Exception:
            return False

    def _results_match(self, wheel_result, classical_result) -> bool:
        """Checks if Wheel and classical results match."""
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
        Tests wheel_calculus on counterexamples from the equations database.

        Verifies the tripartite division:
          REMOVABLE — Wheel=⊥ but analytical limit exists
          BOTTOM    — genuine pole (irremovable ⊥)
          FINITE    — regular point (Wheel OK without interference)

        Returns:
            dictionary with stats: total, correct, removable, poles, finite
        """
        print("\n" + "═" * 62)
        print("  WHEEL CALCULUS — Verification of four-fold singularity division")
        print("  wheel_algebra (⊥) vs wheel_calculus (⊥/lim/order+res)")
        print("═" * 62)

        x, m, p, r, r_s = sp.symbols("x m p r r_s", real=True)
        omega, omega0    = sp.symbols("omega omega0", real=True)

        # ── Counterexamples from database — should be REMOVABLE ──────────
        print("\n  [1/3] Database counterexamples (Wheel=⊥, but limit exists)\n")
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
                "name": "(1 - cos(x))/x²  [lim=1/2, not 1!]",
                "expr": (1 - sp.cos(x)) / x**2,
                "variables": [(x, sp.S.Zero)],
                "expected_limit": sp.Rational(1, 2),
            },
            {
                "name": "Rayleigh-Jeans: ħω/(e^(ħω/kT)-1) as T→∞",
                "expr": x / (sp.exp(x) - 1),           # x = ħω/kT → 0
                "variables": [(x, sp.S.Zero)],
                "expected_limit": sp.S.One,
            },
        ]

        print(f"  {'Equation':<44} {'Type':<12} {'lim':<8} {'OK?'}")
        print(f"  {'─'*44} {'─'*12} {'─'*8} {'─'*4}")
        res_rem = classify_batch(cases_removable, max_order=max_order, verbose=True)
        rem_ok  = sum(1 for r in res_rem if r["correct"] and r["result_type"] == "REMOVABLE")

        # ── Genuine poles — should be POLE with residue ──────────────────
        print("\n  [2/3] Genuine database poles (⊥ — now with order and residue)\n")
        cases_poles = [
            {
                "name": "Scalar propagator 1/(p²-m²) [on-shell]",
                "expr": 1 / (p**2 - m**2),
                "variables": [(p, m)],
                "expected_limit": sp.zoo,
            },
            {
                "name": "Schwarzschild g_rr at r=r_s",
                "expr": 1 / (1 - r_s / r),
                "variables": [(r, r_s)],
                "expected_limit": sp.zoo,
            },
            {
                "name": "Resonance 1/(ω²-ω₀²) at ω=ω₀",
                "expr": 1 / (omega**2 - omega0**2),
                "variables": [(omega, omega0)],
                "expected_limit": sp.zoo,
            },
            {
                "name": "Photon propagator 1/k² at k=0",
                "expr": 1 / x**2,
                "variables": [(x, sp.S.Zero)],
                "expected_limit": sp.zoo,
            },
        ]

        print(f"  {'Equation':<44} {'Type':<14} {'res':<16} {'OK?'}")
        print(f"  {'─'*44} {'─'*14} {'─'*16} {'─'*4}")
        res_pol = classify_batch(cases_poles, max_order=max_order, verbose=True)
        pol_ok  = sum(1 for r in res_pol if r["correct"] and r["result_type"] in ("POLE", "BOTTOM"))

        # ── Regular points — Wheel should not interfere ───────────────────
        print("\n  [3/3] Regular points (Wheel OK, calculus does not interfere)\n")
        cases_reg = [
            {
                "name": "Euclidean KG 1/(p²+m²) at p=0, m=1",
                "expr": 1 / (p**2 + m**2),
                "variables": [(p, sp.S.Zero), (m, sp.S.One)],
                "expected_limit": sp.S.One,
            },
            {
                "name": "Photon effective potential at r=3r_s/2",
                "expr": (1 - r_s / r) / r**2,
                "variables": [(r, sp.Rational(3, 2) * r_s)],
                "expected_limit": None,  # finite, but symbolic
            },
        ]

        print(f"  {'Equation':<44} {'Type':<12} {'lim':<8} {'OK?'}")
        print(f"  {'─'*44} {'─'*12} {'─'*8} {'─'*4}")
        res_reg = classify_batch(cases_reg, max_order=max_order, verbose=True)
        reg_ok  = sum(1 for r in res_reg if r["result_type"] == "FINITE")

        # ── Summary ──────────────────────────────────────────────────────
        total    = len(res_rem) + len(res_pol) + len(res_reg)
        correct  = rem_ok + pol_ok + reg_ok
        score    = correct / total if total else 0.0

        print(f"\n{'─'*62}")
        print(f"  wheel_calculus result:")
        print(f"    Removable singularities : {rem_ok}/{len(res_rem)} correctly → RemovableSingularity")
        print(f"    Poles with residue      : {pol_ok}/{len(res_pol)} correctly → PoleSingularity")
        print(f"    Regular points          : {reg_ok}/{len(res_reg)} correctly → finite")
        print(f"    Total                   : {correct}/{total}  ({score:.0%})")
        print(f"\n{'═'*62}")

        # ── Key example with report ──────────────────────────────────────
        print("\n  Detailed report — sinc(x) (key counterexample):\n")
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
        """Runs the standard consistency test suite for known equations."""

        print("═" * 62)
        print("  CONSISTENCY CHECKER — Standard Test Suite")
        print("═" * 62)

        r, r_s, m, p, a, k_sym = sp.symbols("r r_s m p a k")

        tests = [
            {
                "name":           "Schwarzschild g_rr",
                "expr":           1 / (1 - r_s / r),
                "var":            r,
                "extra_subs":     {r_s: sp.Integer(1)},
                "regular_values": [sp.Rational(2), sp.Rational(3), sp.Rational(5)],
                "singular_values":[sp.Integer(1), sp.Integer(0)],
                "known_singular": [sp.Integer(1)],
            },
            {
                "name":           "Kretschmann invariant K=12r_s²/r⁶",
                "expr":           12 * r_s**2 / r**6,
                "var":            r,
                "extra_subs":     {r_s: sp.Integer(1)},
                "regular_values": [sp.Integer(1), sp.Rational(3, 2), sp.Integer(2)],
                "singular_values":[sp.Integer(0)],
                "known_singular": [sp.Integer(0)],
            },
            {
                "name":           "Scalar propagator 1/(p²-m²)",
                "expr":           1 / (p**2 - m**2),
                "var":            p,
                "extra_subs":     {m: sp.Integer(1)},
                "regular_values": [sp.Integer(0), sp.Rational(1, 2), sp.Rational(3, 2)],
                "singular_values":[sp.Integer(1), sp.Integer(-1)],
                "known_singular": [sp.Integer(1), sp.Integer(-1)],
            },
            {
                "name":           "Friedmann term k·c²/a² (k=c=1)",
                "expr":           k_sym / a**2,
                "var":            a,
                "extra_subs":     {k_sym: sp.Integer(1)},
                "regular_values": [sp.Integer(1), sp.Integer(2), sp.Rational(1, 2)],
                "singular_values":[sp.Integer(0)],
                "known_singular": [sp.Integer(0)],
            },
            {
                "name":           "Expression without singularities: x²+1",
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
        print(f"  GLOBAL RESULT: {avg:.0%}  ({len(tests)} expressions)")
        print(f"{'═'*62}")


if __name__ == "__main__":
    checker = ConsistencyChecker()
    checker.run_standard_suite()